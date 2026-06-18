from __future__ import annotations

from pathlib import Path

from wai_r0.cli import main
from wai_r0.config import ReasonerConfig
from wai_r0.training.language_csv import inspect_language_csv, iter_language_texts, run_csv_language_probe
from wai_r0.training.markdown_plan import load_markdown_training_plan, run_markdown_training_plan


def write_csv(path: Path) -> None:
    path.write_text(
        "text\n"
        "A cat sits on a mat.\n"
        "A dog runs in a yard.\n"
        "A noun names a thing.\n"
        "A verb names an action.\n",
        encoding="utf-8",
    )


def test_inspect_language_csv_detects_text_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "basic.csv"
    write_csv(csv_path)
    inspection = inspect_language_csv(csv_path)
    assert inspection.exists is True
    assert inspection.detected_text_column == "text"
    assert inspection.nonempty_rows == 4
    assert inspection.max_chars > inspection.min_chars


def test_iter_language_texts_streams_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "basic.csv"
    write_csv(csv_path)
    rows = list(iter_language_texts(csv_path, max_rows=2))
    assert rows == ["A cat sits on a mat.", "A dog runs in a yard."]


def test_csv_language_probe_runs_and_can_checkpoint(tmp_path: Path) -> None:
    csv_path = tmp_path / "basic.csv"
    ckpt = tmp_path / "probe.pt"
    write_csv(csv_path)
    cfg = ReasonerConfig(max_seq_len=32, seed=1234)
    result = run_csv_language_probe(
        cfg,
        csv_path,
        steps=1,
        batch_size=2,
        seq_len=24,
        max_rows=4,
        eval_rows=2,
        checkpoint_path=ckpt,
    )
    assert result.rows_consumed == 2
    assert result.eval_loss > 0
    assert result.seq_len == 24
    assert ckpt.exists()


def test_markdown_plan_supports_csv_language_mode(tmp_path: Path) -> None:
    csv_path = tmp_path / "basic.csv"
    plan_path = tmp_path / "train.md"
    write_csv(csv_path)
    plan_path.write_text(
        f"""---
mode: csv_language
config: configs/model/nano.yaml
csv_path: {csv_path}
text_column: text
steps: 1
batch_size: 2
seq_len: 24
max_rows: 4
eval_rows: 2
output: {tmp_path / 'csv_probe'}
---
""",
        encoding="utf-8",
    )
    plan = load_markdown_training_plan(plan_path)
    assert plan.mode == "csv_language"
    assert plan.csv_path == str(csv_path)
    report, loaded = run_markdown_training_plan(plan_path)
    assert loaded.mode == "csv_language"
    assert report.result_type == "v0.4 CSV language-readiness experiment"


def test_cli_train_csv_and_inspect_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "basic.csv"
    out = tmp_path / "report"
    inspect_out = tmp_path / "inspect.json"
    write_csv(csv_path)
    assert main(["inspect-csv", "--csv", str(csv_path), "--output", str(inspect_out)]) == 0
    assert inspect_out.exists()
    assert main([
        "train-csv",
        "--csv",
        str(csv_path),
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--seq-len",
        "24",
        "--max-rows",
        "4",
        "--eval-rows",
        "2",
        "--output",
        str(out),
    ]) == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_legacy_train_csv_normalization() -> None:
    from wai_r0.cli import normalize_legacy_train_args

    assert normalize_legacy_train_args(["-train", "training/basic_language_sample.csv", "--steps", "1"]) == [
        "train-csv",
        "--csv",
        "training/basic_language_sample.csv",
        "--steps",
        "1",
    ]


def test_train_subcommand_accepts_csv(tmp_path) -> None:
    from wai_r0.cli import main

    csv_path = tmp_path / "toy.csv"
    csv_path.write_text("text\nhello world\n", encoding="utf-8")
    output = tmp_path / "csv_train"
    rc = main([
        "train",
        str(csv_path),
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        "8",
        "--max-rows",
        "1",
        "--eval-rows",
        "1",
        "--output",
        str(output),
    ])
    assert rc == 0
    assert (tmp_path / "csv_train.json").exists()
    assert (tmp_path / "csv_train.md").exists()



def test_audit_language_csv_detects_splits_and_duplicates(tmp_path: Path) -> None:
    from wai_r0.training.language_csv import CSVSplitSpec, audit_language_csv

    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "text\n"
        "repeat me\n"
        "repeat me\n"
        "unique one\n"
        "unique two\n"
        "unique three\n",
        encoding="utf-8",
    )
    audit = audit_language_csv(csv_path, split_spec=CSVSplitSpec(train=0.8, val=0.1, test=0.1, seed=42))
    assert audit.nonempty_rows == 5
    assert audit.duplicate_rows == 1
    assert sum(audit.split_counts.values()) == 5
    assert audit.mean_chars > 0


def test_csv_language_probe_writes_log_and_best_checkpoint(tmp_path: Path) -> None:
    from wai_r0.training.language_csv import run_csv_language_probe

    csv_path = tmp_path / "basic.csv"
    write_csv(csv_path)
    ckpt = tmp_path / "probe.pt"
    log = tmp_path / "train.jsonl"
    cfg = ReasonerConfig(max_seq_len=32, seed=1234)
    result = run_csv_language_probe(
        cfg,
        csv_path,
        steps=2,
        batch_size=2,
        seq_len=24,
        max_rows=4,
        eval_rows=2,
        checkpoint_path=ckpt,
        log_path=log,
        eval_interval=1,
        train_fraction=0.75,
        val_fraction=0.25,
        test_fraction=0.0,
    )
    assert ckpt.exists()
    assert log.exists()
    assert result.best_eval_loss is not None
    assert result.audit["split_counts"]["train"] > 0
    assert len(result.history) == 2
    assert result.uniform_baseline_loss is not None
    assert result.unigram_baseline is not None


def test_cli_audit_csv_and_sample_checkpoint(tmp_path: Path) -> None:
    from wai_r0.cli import main

    csv_path = tmp_path / "basic.csv"
    write_csv(csv_path)
    audit_out = tmp_path / "audit.json"
    ckpt = tmp_path / "probe.pt"
    assert main(["audit-csv", "--csv", str(csv_path), "--output", str(audit_out), "--max-rows", "4"]) == 0
    assert audit_out.exists()
    assert main([
        "train-csv",
        "--csv", str(csv_path),
        "--steps", "1",
        "--batch-size", "2",
        "--seq-len", "24",
        "--max-rows", "4",
        "--eval-rows", "2",
        "--checkpoint", str(ckpt),
        "--train-fraction", "0.75",
        "--val-fraction", "0.25",
        "--test-fraction", "0.0",
    ]) == 0
    assert ckpt.exists()
    assert main(["sample-csv", "--checkpoint", str(ckpt), "--prompt", "A", "--max-new-tokens", "2"]) == 0


def test_chat_csv_auto_detects_user_assistant_and_system(tmp_path: Path) -> None:
    from wai_r0.training.language_csv import audit_language_csv

    csv_path = tmp_path / "chat.csv"
    csv_path.write_text(
        "id,split,system,user,assistant\n"
        "1,train,Be concise,hello,Hello. What do you need help with?\n"
        "2,val,Be concise,define benchmark,A benchmark is a test used to compare performance.\n"
        "3,test,Be concise,yo can you help me,Yes. Send the task.\n",
        encoding="utf-8",
    )
    rows = list(iter_language_texts(csv_path, max_rows=3))
    assert rows[0].startswith("SYSTEM:\nBe concise")
    assert "USER:\nhello" in rows[0]
    assert "ASSISTANT:\nHello. What do you need help with?" in rows[0]
    audit = audit_language_csv(csv_path, max_rows=3)
    assert audit.text_column == "user"
    assert audit.target_column == "assistant"
    assert audit.split_counts == {"train": 1, "val": 1, "test": 1}


def test_chat_csv_recovers_from_old_text_column_default(tmp_path: Path) -> None:
    csv_path = tmp_path / "chat.csv"
    csv_path.write_text(
        "id,split,system,user,assistant\n"
        "1,train,Be useful,hello,Hello.\n",
        encoding="utf-8",
    )
    rows = list(iter_language_texts(csv_path, text_column="text", max_rows=1))
    assert rows and "USER:\nhello" in rows[0]
