from __future__ import annotations

import json

import pytest
import yaml

from wai_r0.experiments.runner import run_experiment
from wai_r0.reporting import load_report, write_rendered_report


def _write_config(path, attention: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 96,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_ff": 32,
                "max_seq_len": 32,
                "attention_type": attention,
                "mla_latent_dim": 4,
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )


def test_profile_experiment_writes_auditable_report(tmp_path) -> None:
    _write_config(tmp_path / "candidate.yaml", "mla_lite")
    _write_config(tmp_path / "control.yaml", "gqa")
    manifest = {
        "id": "test-profile",
        "kind": "profile",
        "hypothesis": "compressed cache uses less payload memory",
        "candidate": "mla",
        "control": "gqa",
        "matching_rule": "token_matched",
        "evidence_class": "systems_performance",
        "datasets": ["deterministic-random-v1"],
        "seeds": [1, 2],
        "primary_metric": "kv_cache_reduction",
        "thresholds": {"keep": 0.2, "kill": 0.01, "higher_is_better": True},
        "minimum_successful_seeds": 2,
        "known_confounds": ["tiny CPU test"],
        "execution": {
            "candidate_config": "candidate.yaml",
            "control_config": "control.yaml",
            "sequence_length": 8,
            "warmup_runs": 0,
            "measured_runs": 1,
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    report_path = tmp_path / "report.json"
    report = run_experiment(manifest_path, output=report_path)
    assert report.decision == "keep"
    assert report.identity.experiment_hash
    restored = load_report(report_path)
    assert restored.metrics["paired_comparison"]["count"] == 2
    markdown = write_rendered_report(tmp_path / "report.md", restored)
    html = write_rendered_report(tmp_path / "report.html", restored)
    assert "Decision gates" in markdown.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.read_text(encoding="utf-8")
    json.loads(report_path.read_text(encoding="utf-8"))


def test_external_metric_experiment_can_only_keep_with_passing_gates() -> None:
    report = run_experiment(
        __import__(
            "wai_r0.experiments.manifest", fromlist=["ExperimentManifest"]
        ).ExperimentManifest.from_dict(
            {
                "id": "external-test",
                "hypothesis": "candidate is better",
                "candidate": "candidate",
                "control": "control",
                "matching_rule": "token_matched",
                "evidence_class": "architecture_prior",
                "datasets": ["fixture"],
                "seeds": [1, 2, 3],
                "primary_metric": "score",
                "thresholds": {"keep": 0.05, "kill": -0.05},
                "known_confounds": ["fixture values"],
                "execution": {
                    "candidate_values": [0.8, 0.9, 1.0],
                    "control_values": [0.7, 0.8, 0.9],
                },
            }
        )
    )
    assert report.decision == "keep"
    assert all(gate.status == "pass" for gate in report.gates)


def test_algorithmic_execution_rejects_preregistered_step_budget_overrun(tmp_path) -> None:
    from wai_r0.experiments.manifest import ExperimentManifest
    from wai_r0.experiments.runner import ExperimentExecutionError, run_experiment

    manifest = ExperimentManifest.from_dict(
        {
            "id": "budget-overrun",
            "kind": "algorithmic",
            "hypothesis": "budget must be enforced before training",
            "candidate": "candidate",
            "control": "control",
            "matching_rule": "token_matched",
            "evidence_class": "learned_algorithmic",
            "datasets": ["copy"],
            "seeds": [1],
            "primary_metric": "ood_token_accuracy",
            "thresholds": {"keep": 0.1, "kill": -0.1, "higher_is_better": True},
            "maximum_budget": {"optimizer_steps_per_variant": 2},
            "execution": {"train_steps": 3},
        }
    )
    with pytest.raises(ExperimentExecutionError, match="exceeds"):
        run_experiment(manifest, repository=tmp_path)
