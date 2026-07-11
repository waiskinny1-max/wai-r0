from __future__ import annotations

from pathlib import Path

import yaml

from wai_r0.experiments.runner import run_experiment


def _config(path: Path, *, recurrent_depth: int) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 96,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "n_kv_heads": 4,
                "d_ff": 32,
                "max_seq_len": 32,
                "recurrent_depth": recurrent_depth,
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )


def test_algorithmic_experiment_executes_candidate_and_control(tmp_path) -> None:
    candidate = tmp_path / "candidate.yaml"
    control = tmp_path / "control.yaml"
    _config(candidate, recurrent_depth=2)
    _config(control, recurrent_depth=1)
    manifest = tmp_path / "algorithmic.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "id": "algorithmic-smoke",
                "kind": "algorithmic",
                "hypothesis": "recurrent refinement changes held-out parity accuracy",
                "candidate": "recurrent",
                "control": "dense",
                "matching_rule": "token_matched",
                "evidence_class": "learned_algorithmic",
                "datasets": ["parity"],
                "seeds": [3],
                "primary_metric": "ood_token_accuracy",
                "thresholds": {"keep": 0.2, "kill": -0.2},
                "known_confounds": ["one optimizer step smoke test"],
                "execution": {
                    "candidate_config": "candidate.yaml",
                    "control_config": "control.yaml",
                    "task": "parity",
                    "train_lengths": [3],
                    "id_length": 3,
                    "ood_length": 5,
                    "batch_size": 2,
                    "train_steps": 1,
                    "eval_batches": 1,
                    "learning_rate": 0.001,
                    "candidate_model_mode": "think",
                    "candidate_recurrent_steps": 2,
                    "control_model_mode": "fast",
                },
            }
        ),
        encoding="utf-8",
    )
    report = run_experiment(manifest)
    assert report.decision in {"keep", "kill", "re_test"}
    assert report.metrics["raw"]["task"] == "parity"
    row = report.metrics["raw"]["seeds"][0]
    assert row["candidate"]["training"]["steps"] == 1
    assert row["control"]["training"]["steps"] == 1
    assert report.metrics["paired_comparison"]["count"] == 1
