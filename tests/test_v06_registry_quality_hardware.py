from __future__ import annotations

import time
from pathlib import Path

import torch

from wai_r0.config import ReasonerConfig
from wai_r0.core.reproducibility import canonical_hash
from wai_r0.hardware import calibrate_model, estimate_training_memory, runtime_capabilities
from wai_r0.model import ReasonerCore
from wai_r0.quality import inspect_release
from wai_r0.registry import RunRecord, RunRegistry
from wai_r0.version import __version__


def test_registry_roundtrip_and_artifacts(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.sqlite")
    now = time.time()
    config = {"x": 1}
    record = RunRecord(
        run_id="run-1",
        status="completed",
        created_at=now,
        updated_at=now,
        version=__version__,
        commit=None,
        experiment_hash=None,
        config_hash=canonical_hash(config),
        decision="re_test",
        evidence_class="learned_language",
        manifest_path=None,
        report_path="report.json",
        checkpoint_path=None,
        hardware={"device": "cpu"},
        config=config,
        metrics={"loss": 1.0},
    )
    registry.upsert(record)
    registry.add_artifact("run-1", "report", "report.json", sha256="a" * 64)
    assert registry.get("run-1") == record
    assert registry.list()[0].run_id == "run-1"
    assert registry.artifacts("run-1")["report"]["sha256"] == "a" * 64


def test_release_doctor_reads_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    report = inspect_release(root)
    assert report.version == __version__
    assert any(check.name == "ci_workflow_parse" for check in report.checks)


def test_cpu_hardware_capabilities_and_memory_estimate() -> None:
    capabilities = runtime_capabilities()
    assert capabilities["devices"][0]["device_type"] == "cpu"
    config = ReasonerConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        d_ff=32,
        max_seq_len=32,
    )
    model = ReasonerCore(config)
    estimate = estimate_training_memory(
        model,
        batch_size=2,
        sequence_length=16,
        d_model=config.d_model,
        n_layers=config.n_layers,
    )
    assert estimate.estimated_total_bytes > estimate.parameter_bytes
    calibration = calibrate_model(config)
    assert calibration.attempts == []
    assert calibration.recommended_batch_size is None
    assert torch.cuda.is_available() is False or capabilities["cuda_available"] is True
