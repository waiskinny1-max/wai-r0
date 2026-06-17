from __future__ import annotations

import json
import sys
from pathlib import Path

from wai_r0.gui import (
    CSVTrainGuiOptions,
    build_audit_csv_command,
    build_sample_csv_command,
    build_train_csv_command,
    command_entrypoint,
    parse_training_event,
)


def test_command_entrypoint_prefers_main_py(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    assert command_entrypoint(tmp_path) == [sys.executable, "-u", "main.py"]


def test_train_csv_gui_command_includes_stream_and_budget(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    cmd = build_train_csv_command(
        CSVTrainGuiOptions(
            csv_path="training/data.csv",
            text_column="text",
            target_column="answer",
            steps=123,
            batch_size=7,
            seq_len=96,
            max_rows=500,
            checkpoint="reports/model.pt",
            log="reports/log.jsonl",
            output="reports/out",
        ),
        cwd=tmp_path,
    )
    assert cmd[:3] == [sys.executable, "-u", "main.py"]
    assert "train-csv" in cmd
    assert "--stream" in cmd
    assert cmd[cmd.index("--csv") + 1] == "training/data.csv"
    assert cmd[cmd.index("--target-column") + 1] == "answer"
    assert cmd[cmd.index("--steps") + 1] == "123"
    assert cmd[cmd.index("--batch-size") + 1] == "7"
    assert cmd[cmd.index("--seq-len") + 1] == "96"


def test_audit_and_sample_gui_commands(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    audit = build_audit_csv_command("training/data.csv", max_rows=None, cwd=tmp_path)
    assert audit[:3] == [sys.executable, "-u", "main.py"]
    assert audit[audit.index("--csv") + 1] == "training/data.csv"
    assert "--max-rows" not in audit

    sample = build_sample_csv_command("reports/model.pt", "A noun is", 32, cwd=tmp_path)
    assert sample[:3] == [sys.executable, "-u", "main.py"]
    assert sample[sample.index("--checkpoint") + 1] == "reports/model.pt"
    assert sample[sample.index("--max-new-tokens") + 1] == "32"
    assert "--stream" in sample


def test_parse_training_event() -> None:
    payload = {"step": 4, "train_loss": 3.2, "eval_loss": 3.1, "seconds_elapsed": 2.0}
    parsed = parse_training_event("[train] " + json.dumps(payload))
    assert parsed == payload
    assert parse_training_event("ordinary output") is None
