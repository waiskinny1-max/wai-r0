from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from wai_r0.app.services import LanguageTrainingRequest, run_language_training
from wai_r0.reporting import load_report


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 261,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "n_kv_heads": 4,
                "d_ff": 32,
                "max_seq_len": 64,
                "seed": 5,
            }
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "split",
                "task_family",
                "difficulty",
                "system",
                "user",
                "assistant",
                "answer_format",
                "eval_type",
                "metadata_json",
            ],
        )
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "id": str(index),
                    "split": "train",
                    "task_family": "fixture",
                    "difficulty": "easy",
                    "system": "Be brief.",
                    "user": f"Input {index}",
                    "assistant": f"Output {index}",
                    "answer_format": "text",
                    "eval_type": "exact",
                    "metadata_json": json.dumps({"index": index}),
                }
            )


def test_language_service_writes_complete_artifact_set_and_resumes(tmp_path) -> None:
    config = tmp_path / "model.yaml"
    dataset = tmp_path / "data.csv"
    output = tmp_path / "run"
    _write_config(config)
    _write_csv(dataset)

    first = run_language_training(
        LanguageTrainingRequest(
            model_config=config,
            csv_path=dataset,
            output_dir=output,
            max_steps=1,
            batch_size=2,
            sequence_length=48,
            train_fraction=1.0,
            val_fraction=0.0,
            test_fraction=0.0,
        )
    )
    assert first.checkpoint is not None
    for path in (
        first.report_json,
        first.report_markdown,
        first.report_html,
        first.dataset_manifest,
        first.tokenizer_manifest,
        first.event_log,
        first.checkpoint,
    ):
        assert Path(path).is_file()
    assert Path(first.checkpoint + ".sha256").is_file()
    report = load_report(first.report_json)
    assert report.metrics["progress"]["global_step"] == 1
    assert report.data_manifest is not None
    assert report.tokenizer_manifest is not None

    resumed = run_language_training(
        LanguageTrainingRequest(
            model_config=config,
            csv_path=dataset,
            output_dir=output,
            max_steps=2,
            batch_size=2,
            sequence_length=48,
            train_fraction=1.0,
            val_fraction=0.0,
            test_fraction=0.0,
            resume_from=Path(first.checkpoint),
        )
    )
    resumed_report = load_report(resumed.report_json)
    assert resumed_report.metrics["progress"]["global_step"] == 2
    assert resumed_report.provenance["resumed_from"] == first.checkpoint


def test_language_service_accepts_target_token_budget(tmp_path) -> None:
    config = tmp_path / "model.yaml"
    dataset = tmp_path / "data.csv"
    output = tmp_path / "token-budget"
    _write_config(config)
    _write_csv(dataset)

    artifacts = run_language_training(
        LanguageTrainingRequest(
            model_config=config,
            csv_path=dataset,
            output_dir=output,
            max_target_tokens=1,
            batch_size=2,
            sequence_length=48,
            train_fraction=1.0,
            val_fraction=0.0,
            test_fraction=0.0,
        )
    )
    report = load_report(artifacts.report_json)
    assert report.metrics["progress"]["consumed_tokens"] >= 1
    reproduce = (output / "REPRODUCE.txt").read_text(encoding="utf-8")
    assert "--max-target-tokens 1" in reproduce
    assert "--max-steps" not in reproduce
