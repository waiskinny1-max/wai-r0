from __future__ import annotations

import json
import math
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wai_r0.config import ReasonerConfig
from wai_r0.core.reproducibility import atomic_write_json, atomic_write_text
from wai_r0.data.chat import ByteChatTokenizer
from wai_r0.data.csv_reader import DatasetAudit, audit_conversation_csv
from wai_r0.data.manifest import write_dataset_manifest
from wai_r0.data.splits import SplitSpec
from wai_r0.data.streaming import StatefulCSVBatchStream
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
class LanguageTrainingRequest:
    model_config: Path
    csv_path: Path
    output_dir: Path
    max_steps: int | None = None
    max_target_tokens: int | None = None
    batch_size: int = 8
    sequence_length: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"
    schedule: ScheduleName = "cosine"
    warmup_steps: int = 0
    checkpoint_every: int = 0
    evaluate_every: int = 0
    validation_batches: int = 8
    max_rows: int | None = None
    split_seed: int = 1337
    train_fraction: float = 0.90
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    respect_declared_split: bool = False
    resume_from: Path | None = None
    require_checkpoint_digest: bool = True
    shuffle_buffer_size: int = 256
    shuffle_seed: int | None = None
    pack_sequences: bool = False
    cpu_threads: int | None = None

    def validate(self) -> None:
        if self.max_steps is None and self.max_target_tokens is None:
            raise ValueError("max_steps or max_target_tokens must be set")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when set")
        if self.max_target_tokens is not None and self.max_target_tokens < 1:
            raise ValueError("max_target_tokens must be positive when set")
        if self.batch_size < 1 or self.sequence_length < 2:
            raise ValueError("batch_size must be positive and sequence_length at least 2")
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError("max_rows must be positive when set")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size cannot be negative")
        if self.cpu_threads is not None and (
            isinstance(self.cpu_threads, bool) or self.cpu_threads < 1
        ):
            raise ValueError("cpu_threads must be a positive integer when set")
        SplitSpec(
            train=self.train_fraction,
            val=self.val_fraction,
            test=self.test_fraction,
            seed=self.split_seed,
            respect_declared=self.respect_declared_split,
        ).validate()


@dataclass(frozen=True, slots=True)
class LanguageTrainingArtifacts:
    output_dir: str
    report_json: str
    report_markdown: str
    report_html: str
    dataset_manifest: str
    tokenizer_manifest: str
    event_log: str
    checkpoint: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset_manifest(
    path: str | Path,
    *,
    output: str | Path,
    split_spec: SplitSpec | None = None,
    max_rows: int | None = None,
) -> tuple[DatasetAudit, Path]:
    spec = split_spec or SplitSpec()
    audit = audit_conversation_csv(path, split_spec=spec, max_rows=max_rows)
    manifest = audit.to_manifest(split_spec=spec)
    return audit, write_dataset_manifest(output, manifest)


def run_language_training(request: LanguageTrainingRequest) -> LanguageTrainingArtifacts:
    request.validate()
    request.output_dir.mkdir(parents=True, exist_ok=True)
    config = ReasonerConfig.from_yaml(request.model_config)
    tokenizer = ByteChatTokenizer()
    if config.vocab_size < tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size={config.vocab_size} is smaller than the byte-chat tokenizer "
            f"vocabulary ({tokenizer.vocab_size})"
        )
    if request.sequence_length > config.max_seq_len:
        raise ValueError("sequence_length exceeds model max_seq_len")

    split_spec = SplitSpec(
        train=request.train_fraction,
        val=request.val_fraction,
        test=request.test_fraction,
        seed=request.split_seed,
        respect_declared=request.respect_declared_split,
    )
    dataset_manifest_path = request.output_dir / "dataset-manifest.json"
    audit, _ = build_dataset_manifest(
        request.csv_path,
        output=dataset_manifest_path,
        split_spec=split_spec,
        max_rows=request.max_rows,
    )
    if audit.accepted_rows == 0:
        raise ValueError("dataset audit accepted no rows")
    if audit.rejected_rows:
        raise ValueError(
            f"dataset audit rejected {audit.rejected_rows} malformed or duplicate-ID rows; "
            "correct the dataset before training"
        )
    if audit.cross_split_duplicate_rows:
        raise ValueError("dataset audit found exact duplicate content crossing assigned splits")
    if audit.split_counts.get("train", 0) == 0:
        raise ValueError("dataset has no training rows under the resolved split policy")
    if request.evaluate_every and audit.split_counts.get("val", 0) == 0:
        raise ValueError("evaluation was requested but the validation split is empty")

    tokenizer_manifest = tokenizer.manifest()
    tokenizer_manifest_path = atomic_write_json(
        request.output_dir / "tokenizer-manifest.json", tokenizer_manifest
    )
    train_source = StatefulCSVBatchStream(
        request.csv_path,
        split="train",
        split_spec=split_spec,
        tokenizer=tokenizer,
        batch_size=request.batch_size,
        max_length=request.sequence_length,
        max_rows=request.max_rows,
        repeat=True,
        shuffle_buffer_size=request.shuffle_buffer_size,
        shuffle_seed=request.shuffle_seed,
        pack_sequences=request.pack_sequences,
    )
    validation_source = (
        StatefulCSVBatchStream(
            request.csv_path,
            split="val",
            split_spec=split_spec,
            tokenizer=tokenizer,
            batch_size=request.batch_size,
            max_length=request.sequence_length,
            max_rows=request.max_rows,
            repeat=True,
            shuffle_buffer_size=0,
            pack_sequences=False,
        )
        if request.evaluate_every
        else None
    )
    model = ReasonerCore(config)
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
        save_final_checkpoint=True,
        require_checkpoint_digest=request.require_checkpoint_digest,
        cpu_threads=request.cpu_threads,
    )
    trainer = Trainer(model, trainer_config)
    event_log_path = request.output_dir / "events.jsonl"

    def write_event(metric: TrainingMetrics) -> None:
        payload = json.dumps(asdict(metric), sort_keys=True, allow_nan=False)
        with event_log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")

    if request.resume_from is None:
        event_log_path.unlink(missing_ok=True)
    result = trainer.train(
        train_source,
        resume_from=request.resume_from,
        validation_batches=validation_source,
        event_callback=write_event,
    )
    if not result.metrics:
        raise RuntimeError("training completed without producing any optimizer-step metrics")
    final_metric = result.metrics[-1]
    finite = all(math.isfinite(value) for value in (final_metric.loss, final_metric.gradient_norm))
    gates = [
        GateResult(
            name="training_numerics",
            status="pass" if finite else "fail",
            explanation=(
                "Final loss and gradient norm are finite."
                if finite
                else "Training produced non-finite loss or gradients."
            ),
        ),
        GateResult(
            name="dataset_integrity",
            status="pass",
            explanation="Audit accepted rows and found no cross-split exact duplicates.",
        ),
        GateResult(
            name="checkpoint_written",
            status="pass" if result.final_checkpoint else "fail",
            explanation=(
                f"Final checkpoint: {result.final_checkpoint}"
                if result.final_checkpoint
                else "No final checkpoint was written."
            ),
        ),
    ]
    identity = RunIdentity.create(
        command=["wai-r0", "train", "csv", str(request.csv_path)],
        config={"request": asdict(request), "model": config.to_dict()},
        repository=Path.cwd(),
    )
    report = ResearchReport(
        identity=identity,
        evidence_class="learned_language",
        resolved_config={
            "request": {
                **asdict(request),
                "model_config": str(request.model_config),
                "csv_path": str(request.csv_path),
                "output_dir": str(request.output_dir),
                "resume_from": str(request.resume_from) if request.resume_from else None,
            },
            "model": config.to_dict(),
            "trainer": asdict(trainer_config),
            "split": split_spec.to_dict(),
        },
        metrics={
            "stopped_reason": result.stopped_reason,
            "progress": asdict(result.progress),
            "final_step": asdict(final_metric),
            "history": [asdict(metric) for metric in result.metrics],
            "dataset_audit": audit.to_dict(),
        },
        gates=gates,
        decision="re_test" if finite and result.final_checkpoint else "kill",
        limitations=[
            "A local CSV training run does not establish general reasoning capability.",
            "Byte-level tokenization is a deterministic control, not an efficiency-optimized tokenizer.",
            "Validation metrics are omitted unless evaluate_every is explicitly enabled.",
        ],
        hardware=default_hardware_info(),
        software=default_software_info(),
        data_manifest=audit.to_manifest(split_spec=split_spec).to_dict(),
        tokenizer_manifest=tokenizer_manifest,
        failures=[] if finite else ["Non-finite final training numerics."],
        provenance={
            "dataset_sha256": audit.sha256,
            "model_config": str(request.model_config),
            "resumed_from": str(request.resume_from) if request.resume_from else None,
        },
        artifacts={
            "dataset_manifest": str(dataset_manifest_path),
            "tokenizer_manifest": str(tokenizer_manifest_path),
            "event_log": str(event_log_path),
            "checkpoint": result.final_checkpoint or "",
        },
    )
    report_json = write_report(request.output_dir / "report.json", report)
    report_markdown = write_rendered_report(request.output_dir / "report.md", report)
    report_html = write_rendered_report(request.output_dir / "report.html", report)
    reproduce_command = [
        "wai-r0",
        "train",
        "csv",
        str(request.csv_path),
        "--config",
        str(request.model_config),
        "--output-dir",
        str(request.output_dir),
    ]
    if request.max_steps is not None:
        reproduce_command.extend(["--max-steps", str(request.max_steps)])
    if request.max_target_tokens is not None:
        reproduce_command.extend(["--max-target-tokens", str(request.max_target_tokens)])
    reproduce_command.extend(
        [
            "--batch-size",
            str(request.batch_size),
            "--seq-len",
            str(request.sequence_length),
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
            "--checkpoint-every",
            str(request.checkpoint_every),
            "--evaluate-every",
            str(request.evaluate_every),
            "--validation-batches",
            str(request.validation_batches),
            "--split-seed",
            str(request.split_seed),
            "--train-fraction",
            str(request.train_fraction),
            "--val-fraction",
            str(request.val_fraction),
            "--test-fraction",
            str(request.test_fraction),
            "--shuffle-buffer-size",
            str(request.shuffle_buffer_size),
        ]
    )
    if request.cpu_threads is not None:
        reproduce_command.extend(["--cpu-threads", str(request.cpu_threads)])
    if request.shuffle_seed is not None:
        reproduce_command.extend(["--shuffle-seed", str(request.shuffle_seed)])
    if request.pack_sequences:
        reproduce_command.append("--pack-sequences")
    if request.max_rows is not None:
        reproduce_command.extend(["--max-rows", str(request.max_rows)])
    if request.respect_declared_split:
        reproduce_command.append("--respect-declared-split")
    if not request.require_checkpoint_digest:
        reproduce_command.append("--allow-missing-checkpoint-digest")
    atomic_write_text(
        request.output_dir / "REPRODUCE.txt",
        shlex.join(reproduce_command),
    )
    return LanguageTrainingArtifacts(
        output_dir=str(request.output_dir),
        report_json=str(report_json),
        report_markdown=str(report_markdown),
        report_html=str(report_html),
        dataset_manifest=str(dataset_manifest_path),
        tokenizer_manifest=str(tokenizer_manifest_path),
        event_log=str(event_log_path),
        checkpoint=result.final_checkpoint,
    )


__all__ = [
    "LanguageTrainingArtifacts",
    "LanguageTrainingRequest",
    "build_dataset_manifest",
    "run_language_training",
]
