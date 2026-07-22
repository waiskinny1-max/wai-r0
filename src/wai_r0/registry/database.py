from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wai_r0.core.reproducibility import canonical_hash

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    status: str
    created_at: float
    updated_at: float
    version: str
    commit: str | None
    experiment_hash: str | None
    config_hash: str
    decision: str | None
    evidence_class: str | None
    manifest_path: str | None
    report_path: str | None
    checkpoint_path: str | None
    hardware: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    failure: str | None = None
    parent_run_id: str | None = None

    def validate(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if self.status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise ValueError(f"unsupported run status: {self.status}")
        if self.created_at < 0 or self.updated_at < self.created_at:
            raise ValueError("invalid run timestamps")
        if self.config_hash != canonical_hash(self.config):
            raise ValueError("config_hash does not match config")
        if self.status == "failed" and not self.failure:
            raise ValueError("failed run must include a failure message")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def summary_dict(self) -> dict[str, Any]:
        """Return list-view metadata without duplicating full histories/configs."""

        self.validate()
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "commit": self.commit,
            "experiment_hash": self.experiment_hash,
            "config_hash": self.config_hash,
            "decision": self.decision,
            "evidence_class": self.evidence_class,
            "report_path": self.report_path,
            "checkpoint_path": self.checkpoint_path,
            "failure": self.failure,
            "parent_run_id": self.parent_run_id,
        }


class RunRegistry:
    """Small transactional SQLite registry for local experiment lineage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    version TEXT NOT NULL,
                    commit_sha TEXT,
                    experiment_hash TEXT,
                    config_hash TEXT NOT NULL,
                    decision TEXT,
                    evidence_class TEXT,
                    manifest_path TEXT,
                    report_path TEXT,
                    checkpoint_path TEXT,
                    hardware_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    failure TEXT,
                    parent_run_id TEXT,
                    FOREIGN KEY(parent_run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);
                CREATE INDEX IF NOT EXISTS runs_experiment_idx ON runs(experiment_hash);
                CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs(parent_run_id);
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT,
                    PRIMARY KEY(run_id, name),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                """
            )
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            elif int(existing[0]) != _SCHEMA_VERSION:
                raise RuntimeError("unsupported run-registry schema version")

    @staticmethod
    def _serialize(payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            version=str(row["version"]),
            commit=None if row["commit_sha"] is None else str(row["commit_sha"]),
            experiment_hash=(
                None if row["experiment_hash"] is None else str(row["experiment_hash"])
            ),
            config_hash=str(row["config_hash"]),
            decision=None if row["decision"] is None else str(row["decision"]),
            evidence_class=(None if row["evidence_class"] is None else str(row["evidence_class"])),
            manifest_path=(None if row["manifest_path"] is None else str(row["manifest_path"])),
            report_path=None if row["report_path"] is None else str(row["report_path"]),
            checkpoint_path=(
                None if row["checkpoint_path"] is None else str(row["checkpoint_path"])
            ),
            hardware=dict(json.loads(row["hardware_json"])),
            config=dict(json.loads(row["config_json"])),
            metrics=dict(json.loads(row["metrics_json"])),
            failure=None if row["failure"] is None else str(row["failure"]),
            parent_run_id=(None if row["parent_run_id"] is None else str(row["parent_run_id"])),
        )

    def upsert(self, record: RunRecord) -> None:
        record.validate()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, status, created_at, updated_at, version, commit_sha,
                    experiment_hash, config_hash, decision, evidence_class,
                    manifest_path, report_path, checkpoint_path, hardware_json,
                    config_json, metrics_json, failure, parent_run_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    version=excluded.version,
                    commit_sha=excluded.commit_sha,
                    experiment_hash=excluded.experiment_hash,
                    config_hash=excluded.config_hash,
                    decision=excluded.decision,
                    evidence_class=excluded.evidence_class,
                    manifest_path=excluded.manifest_path,
                    report_path=excluded.report_path,
                    checkpoint_path=excluded.checkpoint_path,
                    hardware_json=excluded.hardware_json,
                    config_json=excluded.config_json,
                    metrics_json=excluded.metrics_json,
                    failure=excluded.failure,
                    parent_run_id=excluded.parent_run_id
                """,
                (
                    record.run_id,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    record.version,
                    record.commit,
                    record.experiment_hash,
                    record.config_hash,
                    record.decision,
                    record.evidence_class,
                    record.manifest_path,
                    record.report_path,
                    record.checkpoint_path,
                    self._serialize(record.hardware),
                    self._serialize(record.config),
                    self._serialize(record.metrics),
                    record.failure,
                    record.parent_run_id,
                ),
            )

    def get(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row_to_record(row)

    def list(self, *, status: str | None = None, limit: int = 100) -> list[RunRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query = "SELECT * FROM runs"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters = (*parameters, limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        failure: str | None = None,
        decision: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        record = self.get(run_id)
        updated = RunRecord(
            **{
                **record.to_dict(),
                "status": status,
                "updated_at": time.time(),
                "failure": failure,
                "decision": decision if decision is not None else record.decision,
                "metrics": dict(metrics) if metrics is not None else record.metrics,
            }
        )
        self.upsert(updated)
        return updated

    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str | Path,
        *,
        sha256: str | None = None,
    ) -> None:
        self.get(run_id)
        if not name.strip():
            raise ValueError("artifact name cannot be empty")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(run_id, name, path, sha256)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(run_id, name) DO UPDATE SET
                    path=excluded.path, sha256=excluded.sha256
                """,
                (run_id, name, str(path), sha256),
            )

    def artifacts(self, run_id: str) -> dict[str, dict[str, str | None]]:
        self.get(run_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, path, sha256 FROM artifacts WHERE run_id = ? ORDER BY name",
                (run_id,),
            ).fetchall()
        return {
            str(row["name"]): {
                "path": str(row["path"]),
                "sha256": None if row["sha256"] is None else str(row["sha256"]),
            }
            for row in rows
        }


__all__ = ["RunRecord", "RunRegistry"]
