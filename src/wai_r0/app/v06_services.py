from __future__ import annotations

import json
import math
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wai_r0.config import ReasonerConfig
from wai_r0.core.reproducibility import atomic_write_text
from wai_r0.data.compiled import (
    CompiledDatasetManifest,
    StatefulCompiledBatchStream,
)
from wai_r0.model import ReasonerCore
from wai_r0.reporting import (
    GateResult,
    ResearchReport,
    RunIdentity,
    default_hardware_info,
    default_software_info,
    write_rendered_report,
    write_report,
)
from wai_r0.training.engine import Trainer, TrainerConfig, TrainingMetrics
from wai_r0.training.schedules import ScheduleName


@dataclass(frozen=True, slots=True)
class CompiledTrainingRequest:
    model_config: Path
    dataset_dir: Path
    output_dir: Path
    max_steps: int | None = None
    max_target_tokens: int | None = None
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"
    schedule: ScheduleName = "cosine"
    warmup_steps: int = 0
    checkpoint_every: int = 0
    evaluate_every: int = 0
    validation_batches: int = 8
    shuffle: bool = True
    shuffle_seed: int = 1337
    pack_sequences: bool = False
    resume_from: Path | None = None
    require_checkpoint_digest: bool = True
    cpu_threads: int | None = None
    activation_checkpointing: bool = False
    compile_model: bool = False
    fused_optimizer: bool = False
    training_stage: str = "pretraining"
    parent_checkpoint: Path | None = None

    def validate(self) -> None:
        if (self.max_steps is None) == (self.max_target_tokens is None):
            raise ValueError("set exactly one of max_steps or max_target_tokens")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_target_tokens is not None and self.max_target_tokens < 1:
            raise ValueError("max_target_tokens must be positive")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if not self.training_stage.strip():
            raise ValueError("training_stage cannot be empty")


@dataclass(frozen=True, slots=True)
class CompiledTrainingArtifacts:
    output_dir: str
    report_json: str
    report_markdown: str
    report_html: str
    event_log: str
    checkpoint: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_compiled_training(request: CompiledTrainingRequest) -> CompiledTrainingArtifacts:
    request.validate()
    request.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = request.dataset_dir / "manifest.json"
    manifest = CompiledDatasetManifest.load(manifest_path)
    config = ReasonerConfig.from_yaml(request.model_config)
    tokenizer_vocab = int(manifest.tokenizer_manifest.get("vocabulary_size", 0))
    if tokenizer_vocab < 1:
        raise ValueError("compiled dataset tokenizer manifest has no vocabulary size")
    if config.vocab_size < tokenizer_vocab:
        raise ValueError(
            f"model vocab_size={config.vocab_size} is smaller than compiled tokenizer "
            f"vocabulary ({tokenizer_vocab})"
        )
    train_source = StatefulCompiledBatchStream(
        request.dataset_dir,
        split="train",
        batch_size=request.batch_size,
        repeat=True,
        shuffle=request.shuffle,
        seed=request.shuffle_seed,
        pack_sequences=request.pack_sequences,
    )
    validation_source = None
    if request.evaluate_every:
        if manifest.splits.get("val") is None or manifest.splits["val"].examples == 0:
            train_source.close()
            raise ValueError("evaluation requested but compiled validation split is empty")
        validation_source = StatefulCompiledBatchStream(
            request.dataset_dir,
            split="val",
            batch_size=request.batch_size,
            repeat=True,
            shuffle=False,
            pack_sequences=False,
        )
    model = ReasonerCore(config)
    manifest_hash = str(manifest.to_dict()["manifest_hash"])
    trainer_config = TrainerConfig(
        max_steps=request.max_steps,
        max_target_tokens=request.max_target_tokens,
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        mixed_precision=request.mixed_precision,
        schedule=request.schedule,
        warmup_steps=request.warmup_steps,
        checkpoint_every=request.checkpoint_every,
        checkpoint_dir=str(request.output_dir / "checkpoints"),
        evaluate_every=request.evaluate_every,
        validation_batches=request.validation_batches,
        require_checkpoint_digest=request.require_checkpoint_digest,
        cpu_threads=request.cpu_threads,
        activation_checkpointing=request.activation_checkpointing,
        compile_model=request.compile_model,
        fused_optimizer=request.fused_optimizer,
        training_stage=request.training_stage,
        parent_checkpoint=(
            str(request.parent_checkpoint) if request.parent_checkpoint is not None else None
        ),
        dataset_manifest_hash=manifest_hash,
        tokenizer_manifest_hash=manifest.tokenizer_manifest_hash,
    )
    trainer = Trainer(model, trainer_config)
    event_log = request.output_dir / "events.jsonl"
    if request.resume_from is None:
        event_log.unlink(missing_ok=True)

    def on_event(metric: TrainingMetrics) -> None:
        with event_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(metric), sort_keys=True, allow_nan=False) + "\n")

    try:
        result = trainer.train(
            train_source,
            resume_from=request.resume_from,
            validation_batches=validation_source,
            event_callback=on_event,
        )
    finally:
        train_source.close()
        if validation_source is not None:
            validation_source.close()
    if not result.metrics:
        raise RuntimeError("compiled training produced no optimizer-step metrics")
    final = result.metrics[-1]
    finite = all(
        math.isfinite(value) for value in (final.loss, final.gradient_norm, final.parameter_norm)
    )
    gates = [
        GateResult(
            "training_numerics",
            "pass" if finite else "fail",
            "Final loss, gradient norm, and parameter norm are finite."
            if finite
            else "Training ended with non-finite numerical telemetry.",
        ),
        GateResult(
            "compiled_dataset_integrity",
            "pass",
            f"Dataset manifest {manifest_hash} was verified before training.",
        ),
        GateResult(
            "checkpoint_written",
            "pass" if result.final_checkpoint else "fail",
            result.final_checkpoint or "No final checkpoint was produced.",
        ),
    ]
    resolved_request = {
        **asdict(request),
        "model_config": str(request.model_config),
        "dataset_dir": str(request.dataset_dir),
        "output_dir": str(request.output_dir),
        "resume_from": str(request.resume_from) if request.resume_from else None,
        "parent_checkpoint": str(request.parent_checkpoint) if request.parent_checkpoint else None,
    }
    identity = RunIdentity.create(
        command=["wai-r0", "train", "compiled", str(request.dataset_dir)],
        config={"request": resolved_request, "model": config.to_dict()},
        repository=Path.cwd(),
    )
    report = ResearchReport(
        identity=identity,
        evidence_class="learned_language",
        resolved_config={
            "request": resolved_request,
            "model": config.to_dict(),
            "trainer": asdict(trainer_config),
            "compiled_dataset": manifest.to_dict(),
        },
        metrics={
            "stopped_reason": result.stopped_reason,
            "progress": asdict(result.progress),
            "final_step": asdict(final),
            "history": [asdict(metric) for metric in result.metrics],
        },
        gates=gates,
        decision="re_test" if finite and result.final_checkpoint else "kill",
        limitations=[
            "A single local training run does not establish general reasoning capability.",
            "GPU conclusions require execution on the declared target GPU.",
            "Tokenizer and compiled-dataset identities are fixed for exact comparisons.",
        ],
        hardware=default_hardware_info(),
        software=default_software_info(),
        data_manifest=manifest.to_dict(),
        tokenizer_manifest=manifest.tokenizer_manifest,
        failures=[] if finite else ["Non-finite final training telemetry."],
        provenance={
            "compiled_dataset_manifest": str(manifest_path),
            "compiled_dataset_manifest_hash": manifest_hash,
            "model_config": str(request.model_config),
            "resumed_from": str(request.resume_from) if request.resume_from else None,
            "parent_checkpoint": (
                str(request.parent_checkpoint) if request.parent_checkpoint else None
            ),
        },
        artifacts={
            "dataset_manifest": str(manifest_path),
            "event_log": str(event_log),
            "checkpoint": result.final_checkpoint or "",
        },
    )
    report_json = write_report(request.output_dir / "report.json", report)
    report_markdown = write_rendered_report(request.output_dir / "report.md", report)
    report_html = write_rendered_report(request.output_dir / "report.html", report)
    command = [
        "wai-r0",
        "train",
        "compiled",
        str(request.dataset_dir),
        "--config",
        str(request.model_config),
        "--output-dir",
        str(request.output_dir),
        "--batch-size",
        str(request.batch_size),
        "--learning-rate",
        str(request.learning_rate),
        "--weight-decay",
        str(request.weight_decay),
        "--gradient-accumulation-steps",
        str(request.gradient_accumulation_steps),
        "--mixed-precision",
        request.mixed_precision,
        "--schedule",
        request.schedule,
        "--warmup-steps",
        str(request.warmup_steps),
        "--shuffle-seed",
        str(request.shuffle_seed),
        "--training-stage",
        request.training_stage,
    ]
    if request.max_steps is not None:
        command.extend(["--max-steps", str(request.max_steps)])
    if request.max_target_tokens is not None:
        command.extend(["--max-target-tokens", str(request.max_target_tokens)])
    if not request.shuffle:
        command.append("--no-shuffle")
    if request.pack_sequences:
        command.append("--pack-sequences")
    if request.activation_checkpointing:
        command.append("--activation-checkpointing")
    if request.compile_model:
        command.append("--compile-model")
    if request.fused_optimizer:
        command.append("--fused-optimizer")
    atomic_write_text(request.output_dir / "REPRODUCE.txt", shlex.join(command))
    return CompiledTrainingArtifacts(
        output_dir=str(request.output_dir),
        report_json=str(report_json),
        report_markdown=str(report_markdown),
        report_html=str(report_html),
        event_log=str(event_log),
        checkpoint=result.final_checkpoint,
    )


__all__ = [
    "CompiledTrainingArtifacts",
    "CompiledTrainingRequest",
    "run_compiled_training",
]
