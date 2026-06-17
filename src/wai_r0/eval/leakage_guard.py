from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json


@dataclass(frozen=True)
class LeakageFinding:
    """Result of checking one task against the local leakage manifest."""

    task_id: str
    task_hash: str
    split: str
    status: str
    path: str
    previous_splits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_task_payload(payload: dict[str, Any]) -> str:
    """Hash task content with stable JSON ordering.

    The hash intentionally ignores the filesystem path and modification time. It is a
    content identity, not a provenance signature.
    """

    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def hash_task_file(path: str | Path) -> tuple[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    task_id = str(payload.get("id", Path(path).stem))
    return task_id, hash_task_payload(payload)


class LeakageGuard:
    """Small local manifest for keeping generated/dev/public tasks separated.

    This is not a cryptographic guarantee against benchmark leakage. It is a practical
    guardrail: if the same task content appears in multiple declared splits, the report
    must say so instead of silently mixing it into evaluation.
    """

    def __init__(self, manifest_path: str | Path = "reports/leakage_manifest.json") -> None:
        self.manifest_path = Path(manifest_path)
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "tasks": {}}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "tasks" not in data or not isinstance(data["tasks"], dict):
            raise ValueError(f"invalid leakage manifest: {self.manifest_path}")
        return data

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def check_file(self, path: str | Path, split: str, register: bool = False) -> LeakageFinding:
        task_id, task_hash = hash_task_file(path)
        existing = self._manifest["tasks"].get(task_hash, {})
        seen_splits = tuple(sorted(existing.get("splits", [])))

        if not seen_splits:
            status = "new"
        elif split in seen_splits:
            status = "known_same_split"
        else:
            status = "cross_split_duplicate"

        if register:
            now = datetime.now(timezone.utc).isoformat()
            record = self._manifest["tasks"].setdefault(
                task_hash,
                {
                    "task_ids": [],
                    "paths": [],
                    "splits": [],
                    "first_seen": now,
                    "last_seen": now,
                },
            )
            if task_id not in record["task_ids"]:
                record["task_ids"].append(task_id)
            clean_path = str(Path(path))
            if clean_path not in record["paths"]:
                record["paths"].append(clean_path)
            if split not in record["splits"]:
                record["splits"].append(split)
            record["last_seen"] = now

        return LeakageFinding(task_id, task_hash, split, status, str(path), seen_splits)

    def scan_directory(
        self,
        tasks_dir: str | Path,
        split: str,
        register: bool = False,
    ) -> list[LeakageFinding]:
        root = Path(tasks_dir)
        if not root.exists():
            raise FileNotFoundError(f"task directory not found: {root}")
        findings = [self.check_file(path, split, register) for path in sorted(root.glob("*.json"))]
        if register:
            self.save()
        return findings

    def summary(self, findings: list[LeakageFinding]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        return {
            "manifest": str(self.manifest_path),
            "total": len(findings),
            "counts": counts,
            "has_cross_split_duplicates": any(f.status == "cross_split_duplicate" for f in findings),
            "findings": [f.to_dict() for f in findings],
        }
