from __future__ import annotations

import time
from pathlib import Path

from wai_r0.core.reproducibility import canonical_hash, file_sha256
from wai_r0.registry.database import RunRecord, RunRegistry
from wai_r0.reporting import load_report
from wai_r0.version import __version__


def register_report(
    registry: RunRegistry,
    report_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    parent_run_id: str | None = None,
) -> RunRecord:
    source = Path(report_path)
    report = load_report(source)
    now = time.time()
    record = RunRecord(
        run_id=report.identity.run_id,
        status="completed" if not report.failures else "failed",
        created_at=now,
        updated_at=now,
        version=__version__,
        commit=report.identity.git_commit,
        experiment_hash=report.identity.experiment_hash,
        config_hash=canonical_hash(report.resolved_config),
        decision=report.decision,
        evidence_class=report.evidence_class,
        manifest_path=report.artifacts.get("manifest") or report.provenance.get("manifest_path"),
        report_path=str(source),
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        hardware=report.hardware,
        config=report.resolved_config,
        metrics=report.metrics,
        failure="; ".join(report.failures) if report.failures else None,
        parent_run_id=parent_run_id,
    )
    registry.upsert(record)
    registry.add_artifact(record.run_id, "report", source, sha256=file_sha256(source))
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        registry.add_artifact(
            record.run_id,
            "checkpoint",
            checkpoint,
            sha256=file_sha256(checkpoint) if checkpoint.is_file() else None,
        )
    return record


__all__ = ["register_report"]
