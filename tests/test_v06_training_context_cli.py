from __future__ import annotations

import json
from pathlib import Path

import torch

from wai_r0.app.cli import main
from wai_r0.app.v06_services import CompiledTrainingRequest, run_compiled_training
from wai_r0.config import ReasonerConfig
from wai_r0.data.compiled import compile_conversation_dataset
from wai_r0.data.splits import SplitSpec
from wai_r0.eval.context import evaluate_context_task
from wai_r0.model import ReasonerCore
from wai_r0.tokenization import ByteTokenizer
from wai_r0.training.checkpoint import CHECKPOINT_FORMAT_VERSION, inspect_checkpoint


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / "chat.csv"
    rows = ["id,split,system,user,assistant"]
    for index in range(24):
        split = "train" if index < 18 else "val"
        rows.append(f"{index},{split},Be concise,question {index},answer {index}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    root = tmp_path / "compiled"
    compile_conversation_dataset(
        csv_path,
        output_dir=root,
        tokenizer=ByteTokenizer(),
        split_spec=SplitSpec(train=0.75, val=0.25, test=0.0, respect_declared=True),
        max_length=48,
    )
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vocab_size: 261",
                "d_model: 16",
                "n_layers: 1",
                "n_heads: 4",
                "n_kv_heads: 4",
                "d_ff: 32",
                "max_seq_len: 48",
                "device: cpu",
                "dtype: float32",
                "seed: 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, root, config_path


def test_compiled_training_writes_v3_checkpoint_and_report(tmp_path: Path) -> None:
    _csv, root, config_path = _dataset(tmp_path)
    output = tmp_path / "run"
    artifacts = run_compiled_training(
        CompiledTrainingRequest(
            model_config=config_path,
            dataset_dir=root,
            output_dir=output,
            max_steps=1,
            batch_size=2,
            evaluate_every=1,
            validation_batches=1,
            cpu_threads=1,
        )
    )
    assert artifacts.checkpoint is not None
    summary = inspect_checkpoint(artifacts.checkpoint)
    assert summary["format_version"] == CHECKPOINT_FORMAT_VERSION == 3
    assert summary["lineage"]["training_stage"] == "pretraining"
    report = json.loads(Path(artifacts.report_json).read_text(encoding="utf-8"))
    final = report["metrics"]["final_step"]
    assert final["raw_tokens"] > 0
    assert final["parameter_norm"] > 0


def test_activation_checkpointing_dense_model_backward() -> None:
    config = ReasonerConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=32,
        max_seq_len=16,
    )
    model = ReasonerCore(config)
    model.transformer.set_gradient_checkpointing(True)
    model.train()
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    output = model(tokens, return_dict=True)
    loss = output.logits.float().mean()  # type: ignore[union-attr]
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_context_evaluation_shapes_and_chance() -> None:
    model = ReasonerCore(
        ReasonerConfig(
            vocab_size=64,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=4,
            d_ff=32,
            max_seq_len=32,
        )
    )
    result = evaluate_context_task(model, task="needle", cases=8, distractors=3)
    assert result.cases == 8
    assert result.context_length == 14
    assert result.chance_accuracy == 1 / 64


def test_v06_cli_tokenizer_compile_and_verify(tmp_path: Path, capsys) -> None:
    csv_path, _root, _config = _dataset(tmp_path)
    tokenizer_path = tmp_path / "bpe.json"
    assert (
        main(
            [
                "tokenizer",
                "train",
                str(csv_path),
                "--output",
                str(tokenizer_path),
                "--vocab-size",
                "270",
                "--max-training-bytes",
                "100000",
            ]
        )
        == 0
    )
    assert tokenizer_path.is_file()
    compiled = tmp_path / "compiled-bpe"
    assert (
        main(
            [
                "data",
                "compile",
                str(csv_path),
                "--output-dir",
                str(compiled),
                "--tokenizer",
                str(tokenizer_path),
                "--max-length",
                "48",
                "--respect-declared-split",
            ]
        )
        == 0
    )
    assert main(["data", "verify", str(compiled)]) == 0
    assert '"valid": true' in capsys.readouterr().out


def test_language_and_generation_eval_cli(tmp_path: Path, capsys) -> None:
    _csv, root, config_path = _dataset(tmp_path)
    output = tmp_path / "run-eval"
    artifacts = run_compiled_training(
        CompiledTrainingRequest(
            model_config=config_path,
            dataset_dir=root,
            output_dir=output,
            max_steps=1,
            batch_size=2,
            evaluate_every=1,
            validation_batches=1,
            cpu_threads=1,
        )
    )
    assert artifacts.checkpoint is not None
    assert (
        main(
            [
                "eval",
                "language",
                str(root),
                "--config",
                str(config_path),
                "--checkpoint",
                str(artifacts.checkpoint),
                "--batch-size",
                "2",
                "--cpu-threads",
                "1",
            ]
        )
        == 0
    )
    language_payload = json.loads(capsys.readouterr().out)
    assert language_payload["full_split"] is True
    assert language_payload["bits_per_byte"] is not None
    assert language_payload["target_tokens"] > 0

    assert (
        main(
            [
                "eval",
                "generation",
                "--config",
                str(config_path),
                "--checkpoint",
                str(artifacts.checkpoint),
                "--prompt",
                "question 1",
                "--samples",
                "2",
                "--max-new-tokens",
                "2",
                "--cpu-threads",
                "1",
            ]
        )
        == 0
    )
    generation_payload = json.loads(capsys.readouterr().out)
    assert generation_payload["diagnostics"]["sequences"] == 2
    assert len(generation_payload["texts"]) == 2


def test_exact_resume_may_add_parent_lineage(tmp_path: Path) -> None:
    """A continuation may record its source checkpoint without changing training semantics."""
    from wai_r0.config import ReasonerConfig
    from wai_r0.data.compiled import StatefulCompiledBatchStream, compile_conversation_dataset
    from wai_r0.tokenization.byte import ByteTokenizer
    from wai_r0.training.engine import Trainer, TrainerConfig

    csv_path = tmp_path / "chat.csv"
    csv_path.write_text(
        "id,split,system,user,assistant\n"
        "1,train,Be concise,hello,Hello.\n"
        "2,train,Be concise,repeat alpha,alpha\n"
        "3,val,Be concise,hello,Hello.\n",
        encoding="utf-8",
    )
    tokenizer = ByteTokenizer()
    dataset = tmp_path / "dataset"
    compile_conversation_dataset(
        csv_path,
        output_dir=dataset,
        tokenizer=tokenizer,
        max_length=64,
        allow_cross_split_duplicates=True,
    )
    cfg = ReasonerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        d_ff=32,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
        seed=9,
    )
    first_source = StatefulCompiledBatchStream(dataset, batch_size=1, seed=9)
    first = Trainer(
        ReasonerCore(cfg),
        TrainerConfig(
            max_steps=1,
            checkpoint_dir=str(tmp_path / "first"),
            cpu_threads=1,
            dataset_manifest_hash=first_source.dataset.manifest.to_dict()["manifest_hash"],
            tokenizer_manifest_hash=first_source.dataset.manifest.tokenizer_manifest_hash,
        ),
    )
    first_result = first.train(first_source)
    assert first_result.final_checkpoint is not None
    first_source.close()

    second_source = StatefulCompiledBatchStream(dataset, batch_size=1, seed=9)
    second = Trainer(
        ReasonerCore(cfg),
        TrainerConfig(
            max_steps=2,
            checkpoint_dir=str(tmp_path / "second"),
            cpu_threads=1,
            parent_checkpoint=first_result.final_checkpoint,
            dataset_manifest_hash=second_source.dataset.manifest.to_dict()["manifest_hash"],
            tokenizer_manifest_hash=second_source.dataset.manifest.tokenizer_manifest_hash,
        ),
    )
    second_result = second.train(second_source, resume_from=first_result.final_checkpoint)
    second_source.close()
    assert second_result.progress.global_step == 2
    assert second_result.final_checkpoint is not None
    inspected = inspect_checkpoint(second_result.final_checkpoint)
    assert inspected["lineage"]["parent_checkpoint"] == first_result.final_checkpoint
