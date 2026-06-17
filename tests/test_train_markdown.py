from __future__ import annotations

import json
from pathlib import Path

import pytest

from wai_r0.cli import main, normalize_legacy_train_args
from wai_r0.training.markdown_plan import load_markdown_training_plan


def test_load_training_plan_from_yaml_block(tmp_path: Path) -> None:
    plan_path = tmp_path / "training.md"
    plan_path.write_text(
        """# Probe\n\n```yaml\nmode: tiny_probe\nconfig: configs/model/nano.yaml\ntask: reverse\nexamples: 4\nbatch_size: 2\ntrain_len: 4\neval_lens: [4, 8]\n```\n""",
        encoding="utf-8",
    )

    plan = load_markdown_training_plan(plan_path)

    assert plan.task == "reverse"
    assert plan.examples == 4
    assert plan.batch_size == 2
    assert plan.train_len == 4
    assert plan.eval_lens == (4, 8)


def test_load_training_plan_from_key_value_markdown(tmp_path: Path) -> None:
    plan_path = tmp_path / "training.md"
    plan_path.write_text(
        """# Probe\n\n- task: parity\n- examples: 6\n- batch_size: 2\n- train_len: 5\n- eval_lens: 5,10\n""",
        encoding="utf-8",
    )

    plan = load_markdown_training_plan(plan_path)

    assert plan.task == "parity"
    assert plan.eval_lens == (5, 10)


def test_training_plan_rejects_unknown_keys(tmp_path: Path) -> None:
    plan_path = tmp_path / "training.md"
    plan_path.write_text(
        """---\ntask: copy\nshell: rm -rf /\n---\n""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported training plan keys"):
        load_markdown_training_plan(plan_path)


def test_legacy_train_alias_is_normalized() -> None:
    assert normalize_legacy_train_args(["-train", "training.md"]) == ["train", "training.md"]
    assert normalize_legacy_train_args(["--train", "training.md", "--output", "reports/x"]) == [
        "train",
        "training.md",
        "--output",
        "reports/x",
    ]


def test_train_markdown_cli_writes_report(tmp_path: Path) -> None:
    plan_path = tmp_path / "training.md"
    output_stem = tmp_path / "train_report"
    plan_path.write_text(
        f"""# Probe\n\n```yaml\nmode: tiny_probe\nconfig: configs/model/nano.yaml\ntask: copy\nexamples: 4\nbatch_size: 2\ntrain_len: 4\neval_lens: [4]\noutput: {output_stem}\n```\n""",
        encoding="utf-8",
    )

    assert main(["-train", str(plan_path)]) == 0

    json_path = output_stem.with_suffix(".json")
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["name"] == "train_md"
    assert data["benchmark_config"]["training_plan"]["task"] == "copy"
    assert data["raw_metrics"]["training_plan_mode"] == "tiny_probe"
