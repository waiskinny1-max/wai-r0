from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

from wai_r0.v05_cli import main
from wai_r0.version import __version__


def _model_config(path: Path, *, attention: str = "gqa") -> None:
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
                "seed": 2,
            }
        ),
        encoding="utf-8",
    )


def _dataset(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "system", "user", "assistant", "metadata_json"],
        )
        writer.writeheader()
        for index in range(6):
            writer.writerow(
                {
                    "id": str(index),
                    "system": "brief",
                    "user": f"q{index}",
                    "assistant": f"a{index}",
                    "metadata_json": "{}",
                }
            )


def test_native_cli_read_only_commands(tmp_path, capsys) -> None:
    config = tmp_path / "model.yaml"
    dataset = tmp_path / "data.csv"
    _model_config(config)
    _dataset(dataset)

    assert main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    audit_path = tmp_path / "audit.json"
    assert (
        main(
            [
                "data",
                "audit",
                str(dataset),
                "--output",
                str(audit_path),
                "--train-fraction",
                "1",
                "--val-fraction",
                "0",
                "--test-fraction",
                "0",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert audit_path.is_file()

    manifest_path = tmp_path / "dataset-manifest.json"
    assert (
        main(
            [
                "data",
                "manifest",
                str(dataset),
                "--output",
                str(manifest_path),
                "--train-fraction",
                "1",
                "--val-fraction",
                "0",
                "--test-fraction",
                "0",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert manifest_path.is_file()

    assert main(["model", "inspect", "--config", str(config), "--seq-len", "4"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["finite"] is True

    profile_path = tmp_path / "profile.json"
    assert (
        main(
            [
                "profile",
                "--config",
                str(config),
                "--seq-len",
                "4",
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
                "--output",
                str(profile_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert profile_path.is_file()


def test_cli_experiment_report_and_reproduction(tmp_path, capsys) -> None:
    candidate = tmp_path / "candidate.yaml"
    control = tmp_path / "control.yaml"
    _model_config(candidate, attention="mla_lite")
    _model_config(control, attention="gqa")
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "id": "cli-profile",
                "kind": "profile",
                "hypothesis": "latent cache is smaller",
                "candidate": "mla",
                "control": "gqa",
                "matching_rule": "token_matched",
                "evidence_class": "systems_performance",
                "datasets": ["tokens"],
                "seeds": [1],
                "primary_metric": "kv_cache_reduction",
                "thresholds": {"keep": 0.1, "kill": 0.0},
                "known_confounds": ["single CPU seed"],
                "execution": {
                    "candidate_config": "candidate.yaml",
                    "control_config": "control.yaml",
                    "sequence_length": 4,
                    "warmup_runs": 0,
                    "measured_runs": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(["experiment", "validate", str(manifest)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is True

    report = tmp_path / "experiment-report.json"
    assert (
        main(
            [
                "experiment",
                "run",
                str(manifest),
                "--output",
                str(report),
                "--repository",
                str(tmp_path),
                "--render",
                "both",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "keep"
    assert report.is_file()
    assert report.with_suffix(".md").is_file()
    assert report.with_suffix(".html").is_file()

    assert main(["report", "validate", str(report)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    rendered = tmp_path / "manual.md"
    assert main(["report", "render", str(report), "--output", str(rendered)]) == 0
    capsys.readouterr()
    assert rendered.is_file()

    assert main(["reproduce", str(report)]) == 0
    reproduction = json.loads(capsys.readouterr().out)
    assert reproduction["report_valid"] is True
    assert reproduction["executed"] is False

    reproduced = tmp_path / "reproduced.json"
    assert (
        main(
            [
                "reproduce",
                str(report),
                "--execute",
                "--output",
                str(reproduced),
            ]
        )
        == 0
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["executed"] is True
    assert replay["decision_matches"] is True
    assert reproduced.is_file()


def test_main_py_bootstraps_uninstalled_source_layout() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repository / "main.py"), "version"],
        cwd=repository.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == __version__
