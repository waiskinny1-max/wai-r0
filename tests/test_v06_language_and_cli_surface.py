from __future__ import annotations

from pathlib import Path

import torch

from wai_r0.app.cli import main
from wai_r0.config import ReasonerConfig
from wai_r0.data.chat import ByteChatTokenizer, ChatExample, encode_chat_example, pad_chat_batch
from wai_r0.eval.language import evaluate_language_batches
from wai_r0.model import ReasonerCore
from wai_r0.reporting import GateResult, ResearchReport, RunIdentity, write_report


def _config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "vocab_size: 261",
                "d_model: 16",
                "n_layers: 1",
                "n_heads: 4",
                "n_kv_heads: 4",
                "d_ff: 32",
                "max_seq_len: 32",
                "device: cpu",
                "dtype: float32",
                "seed: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_language_evaluation_on_encoded_batch() -> None:
    tokenizer = ByteChatTokenizer()
    examples = [
        encode_chat_example(
            ChatExample(system="", user="hello", assistant="hi"),
            tokenizer=tokenizer,
            max_length=32,
        ),
        encode_chat_example(
            ChatExample(system="", user="bye", assistant="later"),
            tokenizer=tokenizer,
            max_length=32,
        ),
    ]
    input_ids, labels, attention_mask = pad_chat_batch(examples)
    model = ReasonerCore(
        ReasonerConfig(
            vocab_size=261,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=4,
            d_ff=32,
            max_seq_len=32,
        )
    )
    result = evaluate_language_batches(
        model,
        [{"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}],
        max_batches=1,
        raw_bytes=10,
    )
    assert result.batches == 1
    assert result.target_tokens > 0
    assert result.mean_loss > 0
    assert result.bits_per_byte is not None


def test_cli_hardware_release_infer_eval_and_registry(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "model.yaml"
    _config(config_path)

    assert main(["hardware", "inspect"]) == 0
    assert (
        main(
            [
                "hardware",
                "estimate",
                "--config",
                str(config_path),
                "--batch-size",
                "1",
                "--seq-len",
                "8",
            ]
        )
        == 0
    )
    root = Path(__file__).resolve().parents[1]
    assert main(["release", "doctor", "--repository", str(root)]) in {0, 1}
    assert (
        main(
            [
                "infer",
                "generate",
                "--config",
                str(config_path),
                "--prompt",
                "hello",
                "--max-new-tokens",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval",
                "context",
                "--config",
                str(config_path),
                "--task",
                "induction",
                "--cases",
                "4",
                "--distractors",
                "2",
            ]
        )
        == 0
    )

    database = tmp_path / "runs.sqlite"
    assert main(["runs", "init", "--database", str(database)]) == 0
    assert main(["runs", "list", "--database", str(database)]) == 0

    report_path = tmp_path / "report.json"
    report = ResearchReport(
        identity=RunIdentity.create(command=["test"], config={"x": 1}),
        evidence_class="learned_language",
        resolved_config={"x": 1},
        metrics={"loss": 1.0},
        gates=[GateResult("ok", "pass", "passed")],
        decision="re_test",
        limitations=["test report"],
    )
    write_report(report_path, report)
    assert (
        main(
            [
                "runs",
                "register",
                str(report_path),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "runs",
                "show",
                report.identity.run_id,
                "--database",
                str(database),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"run_id"' in output
    assert '"estimated_total_bytes"' in output
    assert torch.isfinite(torch.tensor(1.0))
