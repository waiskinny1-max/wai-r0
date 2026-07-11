from __future__ import annotations

import json

import pytest

from wai_r0.experiments.manifest import ExperimentManifest
from wai_r0.reporting.schema import (
    GateResult,
    ResearchReport,
    RunIdentity,
    load_report,
    write_report,
)


def _manifest() -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "id": "mla-memory-001",
            "hypothesis": "MLA-lite reduces measured cache bytes.",
            "candidate": "mla_lite",
            "control": "gqa",
            "matching_rule": "parameter_matched",
            "evidence_class": "systems_performance",
            "datasets": ["generated-copy-v1"],
            "seeds": [7, 11, 13],
            "primary_metric": "kv_cache_reduction",
            "thresholds": {"keep": 0.30, "kill": 0.05, "higher_is_better": True},
            "secondary_metrics": ["decode_latency_ms"],
            "failure_metrics": ["non_finite_logits"],
            "maximum_budget": {"gpu_hours": 1},
            "known_confounds": ["small model scale"],
        }
    )


def test_manifest_hash_is_stable_and_decision_is_explicit() -> None:
    first = _manifest()
    second = _manifest()
    assert first.manifest_hash == second.manifest_hash
    assert first.thresholds.decide(0.4) == "keep"
    assert first.thresholds.decide(0.01) == "kill"
    assert first.thresholds.decide(0.2) == "re_test"


def test_report_round_trip(tmp_path) -> None:
    identity = RunIdentity.create(command=["wai-r0", "profile"], config={"a": 1})
    report = ResearchReport(
        identity=identity,
        evidence_class="systems_performance",
        resolved_config={"a": 1},
        metrics={"kv_cache_reduction": 0.35},
        gates=[GateResult("correctness", "pass", "cached logits matched")],
        decision="keep",
        limitations=["Tiny model evidence does not establish scale transfer."],
    )
    path = write_report(tmp_path / "report.json", report)
    restored = load_report(path)
    assert restored.identity.run_id == identity.run_id
    assert restored.decision == "keep"
    json.loads(path.read_text(encoding="utf-8"))


def test_failed_gate_cannot_be_overridden_by_blended_score() -> None:
    report = ResearchReport(
        identity=RunIdentity.create(command=["test"], config={}),
        evidence_class="architecture_prior",
        resolved_config={},
        metrics={"score": 0.99},
        gates=[GateResult("correctness", "fail", "non-finite output")],
        decision="keep",
        limitations=["Test report."],
    )
    with pytest.raises(ValueError, match="failed gate"):
        report.validate()
