from __future__ import annotations

import json
from pathlib import Path

from wai_r0.benchmarks import architecture_priors
from wai_r0.config import ReasonerConfig
from wai_r0.eval.prior_diagnostics import run_prior_diagnostics
from wai_r0.eval.suite import run_suite


def test_architecture_prior_diagnostics_report_shape() -> None:
    cfg = ReasonerConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, n_kv_heads=2, d_ff=16, max_seq_len=8)
    report = architecture_priors(cfg, batch_size=1, seq_len=4, recurrent_depths=(1, 2))

    assert report.name == "architecture_priors"
    assert report.result_type == "architecture-prior diagnostic"
    assert 0.0 <= report.raw_metrics["aggregate_prior_score"] <= 1.0
    assert {probe["name"] for probe in report.raw_metrics["probes"]} >= {
        "activation_sanity",
        "positional_addressing",
        "identity_signal_preservation",
        "memory_mechanics",
        "recurrent_consistency",
        "routing_health",
    }


def test_prior_diagnostics_does_not_require_moe() -> None:
    cfg = ReasonerConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, n_kv_heads=2, d_ff=16, max_seq_len=8)
    result = run_prior_diagnostics(cfg, batch_size=1, seq_len=4, recurrent_depths=(1, 2))
    routing = next(probe for probe in result["probes"] if probe["name"] == "routing_health")

    assert routing["metrics"]["evaluated"] is False
    assert routing["score"] == 0.5


def test_prior_diagnostics_evaluates_moe_config() -> None:
    cfg = ReasonerConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, n_kv_heads=2, d_ff=16, max_seq_len=8, use_moe=True, n_experts=2)
    result = run_prior_diagnostics(cfg, batch_size=1, seq_len=4, recurrent_depths=(1, 2))
    routing = next(probe for probe in result["probes"] if probe["name"] == "routing_health")

    assert routing["metrics"]["evaluated"] is True
    assert 0.0 <= routing["score"] <= 1.0


def test_suite_runner_writes_ordered_reports(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
output_dir: {out}
steps:
  - name: zero
    seq_len: 4
  - name: prior
    seq_len: 8
    recurrent_depths: [1, 2]
""".format(out=tmp_path / "reports"),
        encoding="utf-8",
    )
    cfg = ReasonerConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, n_kv_heads=2, d_ff=16, max_seq_len=8)
    result = run_suite(cfg, suite)

    assert [report.name for report in result.reports] == ["zero_neural", "architecture_priors"]
    for json_path, md_path in result.written:
        assert json_path.exists()
        assert md_path.exists()
        assert json.loads(json_path.read_text(encoding="utf-8"))["name"]
