from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from wai_r0.core.runtime import temporary_torch_threads
from wai_r0.model import ModelOutput
from wai_r0.training.checkpoint import (
    RestoredCheckpoint,
    TrainingProgress,
    load_checkpoint,
    save_checkpoint,
)
from wai_r0.training.losses import causal_language_model_loss
from wai_r0.training.optimizer import build_adamw
from wai_r0.training.schedules import ScheduleName, build_scheduler


class StatefulBatchSource(Protocol):
    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, payload: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    max_steps: int | None = None
    max_target_tokens: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    mixed_precision: str = "none"
    schedule: ScheduleName = "cosine"
    warmup_steps: int = 0
    minimum_learning_rate_ratio: float = 0.1
    checkpoint_every: int = 0
    checkpoint_dir: str = "checkpoints"
    evaluate_every: int = 0
    validation_batches: int = 8
    detect_anomaly: bool = False
    save_on_interrupt: bool = True
    save_final_checkpoint: bool = True
    final_checkpoint_name: str = "final.pt"
    model_mode: str = "fast"
    recurrent_steps: int | None = None
    require_checkpoint_digest: bool = True
    cpu_threads: int | None = None

    def validate(self) -> None:
        if self.max_steps is None and self.max_target_tokens is None:
            raise ValueError("max_steps or max_target_tokens must be set")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when set")
        if self.max_target_tokens is not None and self.max_target_tokens < 1:
            raise ValueError("max_target_tokens must be positive when set")
        if self.learning_rate <= 0 or self.adam_epsilon <= 0:
            raise ValueError("learning_rate and adam_epsilon must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if self.schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("unsupported learning-rate schedule")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.max_steps is not None and self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if not 0 <= self.minimum_learning_rate_ratio <= 1:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")
        if self.checkpoint_every < 0 or self.evaluate_every < 0:
            raise ValueError("checkpoint/evaluation intervals cannot be negative")
        if self.validation_batches < 1:
            raise ValueError("validation_batches must be positive")
        if not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir cannot be empty")
        if (
            not self.final_checkpoint_name.strip()
            or Path(self.final_checkpoint_name).name != self.final_checkpoint_name
        ):
            raise ValueError("final_checkpoint_name must be a non-empty file name")
        if self.model_mode not in {"fast", "think"}:
            raise ValueError("model_mode must be fast or think")
        if self.recurrent_steps is not None and self.recurrent_steps < 1:
            raise ValueError("recurrent_steps must be positive when set")
        if self.model_mode == "fast" and self.recurrent_steps is not None:
            raise ValueError("recurrent_steps is only valid in think mode")
        if self.cpu_threads is not None and (
            isinstance(self.cpu_threads, bool) or self.cpu_threads < 1
        ):
            raise ValueError("cpu_threads must be a positive integer when set")


@dataclass(slots=True)
class TrainingMetrics:
    step: int
    loss: float
    language_model_loss: float
    auxiliary_loss: float
    gradient_norm: float
    learning_rate: float
    target_tokens: int
    consumed_tokens: int
    examples: int
    consumed_examples: int
    step_time_ms: float
    target_tokens_per_second: float
    validation_loss: float | None = None


@dataclass(slots=True)
class TrainingResult:
    progress: TrainingProgress
    metrics: list[TrainingMetrics] = field(default_factory=list)
    stopped_reason: str = "budget_reached"
    final_checkpoint: str | None = None


class TrainingOOMError(RuntimeError):
    pass


class Trainer:
    """Explicit local trainer with resumable optimizer, scheduler, RNG, and data state."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = optimizer or build_adamw(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            epsilon=config.adam_epsilon,
        )
        scheduler_steps = config.max_steps or max(1, config.warmup_steps + 1)
        resolved_schedule: ScheduleName = (
            config.schedule if config.max_steps is not None else "constant"
        )
        self.scheduler = scheduler or build_scheduler(
            self.optimizer,
            total_steps=scheduler_steps,
            warmup_steps=min(config.warmup_steps, scheduler_steps - 1),
            schedule=resolved_schedule,
            minimum_ratio=config.minimum_learning_rate_ratio,
        )
        fp16_enabled = config.mixed_precision == "fp16" and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=fp16_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=fp16_enabled)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.config.mixed_precision == "none":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        if self.device.type == "cpu" and dtype == torch.float16:
            raise RuntimeError("FP16 autocast is not supported by this CPU training path")
        return cast(
            contextlib.AbstractContextManager[Any],
            torch.autocast(device_type=self.device.type, dtype=dtype),
        )

    def _resolved_checkpoint_config(self) -> dict[str, Any]:
        model_config = getattr(self.model, "cfg", None)
        if model_config is None and hasattr(self.model, "transformer"):
            model_config = getattr(self.model.transformer, "cfg", None)
        model_payload = (
            model_config.to_dict()
            if model_config is not None and hasattr(model_config, "to_dict")
            else {}
        )
        return {"trainer": asdict(self.config), "model": model_payload}

    @staticmethod
    def _source_state(batches: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, Any]:
        method = getattr(batches, "state_dict", None)
        return method() if callable(method) else {}

    @staticmethod
    def _restore_source_state(
        batches: Iterable[Mapping[str, torch.Tensor]], payload: Mapping[str, Any]
    ) -> None:
        if not payload:
            return
        method = getattr(batches, "load_state_dict", None)
        if not callable(method):
            raise ValueError("checkpoint contains data state but batch source is not stateful")
        method(payload)

    def _validate_resume_config(self, restored: RestoredCheckpoint) -> None:
        if not restored.config:
            return
        current = self._resolved_checkpoint_config()
        saved_model = restored.config.get("model")
        current_model = current.get("model")
        if saved_model != current_model:
            raise ValueError("checkpoint model configuration does not match the current model")

        saved_trainer = restored.config.get("trainer")
        current_trainer = current.get("trainer")
        if not isinstance(saved_trainer, Mapping) or not isinstance(current_trainer, Mapping):
            raise ValueError("checkpoint trainer configuration is missing or malformed")
        mutable_resume_fields = {
            "max_steps",
            "max_target_tokens",
            "checkpoint_every",
            "checkpoint_dir",
            "save_on_interrupt",
            "save_final_checkpoint",
            "final_checkpoint_name",
            "require_checkpoint_digest",
        }
        saved_immutable = {
            key: value for key, value in saved_trainer.items() if key not in mutable_resume_fields
        }
        current_immutable = {
            key: value for key, value in current_trainer.items() if key not in mutable_resume_fields
        }
        if saved_immutable != current_immutable:
            differing = sorted(
                key
                for key in set(saved_immutable) | set(current_immutable)
                if saved_immutable.get(key) != current_immutable.get(key)
            )
            raise ValueError(
                "checkpoint trainer configuration differs in resume-critical fields: "
                + ", ".join(differing)
            )

    def resume(
        self,
        checkpoint: str | Path,
        *,
        batches: Iterable[Mapping[str, torch.Tensor]] | None = None,
        require_digest: bool = False,
    ) -> RestoredCheckpoint:
        restored = load_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
            require_digest=require_digest,
        )
        self._validate_resume_config(restored)
        if batches is not None:
            self._restore_source_state(batches, restored.data_state)
        return restored

    def evaluate(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        max_batches: int | None = None,
    ) -> float:
        limit = max_batches or self.config.validation_batches
        if limit < 1:
            raise ValueError("max_batches must be positive")
        iterator = iter(batches)
        losses: list[float] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for _ in range(limit):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    moved = self._move_batch(batch)
                    with self._autocast():
                        output = self._model_forward(moved)
                        loss, _ = causal_language_model_loss(
                            output.logits,
                            moved["labels"],
                            auxiliary_losses=output.auxiliary_losses,
                        )
                    losses.append(float(loss.detach().float().cpu()))
        finally:
            self.model.train(was_training)
        if not losses:
            raise RuntimeError("validation source yielded no batches")
        return sum(losses) / len(losses)

    def train(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        progress: TrainingProgress | None = None,
        resume_from: str | Path | None = None,
        validation_batches: Iterable[Mapping[str, torch.Tensor]] | None = None,
        event_callback: Callable[[TrainingMetrics], None] | None = None,
    ) -> TrainingResult:
        threads = self.config.cpu_threads if self.device.type == "cpu" else None
        with temporary_torch_threads(threads):
            return self._train_impl(
                batches,
                progress=progress,
                resume_from=resume_from,
                validation_batches=validation_batches,
                event_callback=event_callback,
            )

    def _train_impl(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        progress: TrainingProgress | None = None,
        resume_from: str | Path | None = None,
        validation_batches: Iterable[Mapping[str, torch.Tensor]] | None = None,
        event_callback: Callable[[TrainingMetrics], None] | None = None,
    ) -> TrainingResult:
        current = progress or TrainingProgress()
        if resume_from is not None:
            if progress is not None:
                raise ValueError("progress and resume_from cannot both be supplied")
            current = self.resume(
                resume_from,
                batches=batches,
                require_digest=self.config.require_checkpoint_digest,
            ).progress
        current.validate()
        iterator = iter(batches)
        result = TrainingResult(progress=current)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        training_started = time.perf_counter()

        try:
            while self._within_budget(current):
                metric = self._training_step(iterator, current)
                if (
                    validation_batches is not None
                    and self.config.evaluate_every
                    and current.global_step % self.config.evaluate_every == 0
                ):
                    metric.validation_loss = self.evaluate(validation_batches)
                    best = current.best_metrics.get("validation_loss")
                    if best is None or metric.validation_loss < best:
                        current.best_metrics["validation_loss"] = metric.validation_loss
                result.metrics.append(metric)
                if event_callback is not None:
                    event_callback(metric)
                current.elapsed_seconds += time.perf_counter() - training_started
                training_started = time.perf_counter()
                self._maybe_checkpoint(current, batches)
            current.elapsed_seconds += time.perf_counter() - training_started
            training_started = time.perf_counter()
            if self.config.save_final_checkpoint:
                path = Path(self.config.checkpoint_dir) / self.config.final_checkpoint_name
                self._save_checkpoint(path, current, batches, overwrite=True)
                result.final_checkpoint = str(path)
            return result
        except KeyboardInterrupt:
            result.stopped_reason = "interrupted"
            current.elapsed_seconds += time.perf_counter() - training_started
            training_started = time.perf_counter()
            if self.config.save_on_interrupt:
                path = Path(self.config.checkpoint_dir) / "interrupted.pt"
                self._save_checkpoint(path, current, batches, overwrite=True)
                result.final_checkpoint = str(path)
            return result
        finally:
            current.elapsed_seconds += time.perf_counter() - training_started

    def _within_budget(self, progress: TrainingProgress) -> bool:
        if self.config.max_steps is not None and progress.global_step >= self.config.max_steps:
            return False
        return not (
            self.config.max_target_tokens is not None
            and progress.consumed_tokens >= self.config.max_target_tokens
        )

    def _training_step(
        self,
        iterator: Iterator[Mapping[str, torch.Tensor]],
        progress: TrainingProgress,
    ) -> TrainingMetrics:
        started = time.perf_counter_ns()
        accumulated_target_tokens = 0
        accumulated_examples = 0
        accumulated_total_loss = 0.0
        accumulated_language_loss = 0.0
        accumulated_auxiliary_loss = 0.0

        try:
            anomaly_context = (
                torch.autograd.detect_anomaly()
                if self.config.detect_anomaly
                else contextlib.nullcontext()
            )
            with anomaly_context:
                for _ in range(self.config.gradient_accumulation_steps):
                    batch = self._next_batch(iterator)
                    moved = self._move_batch(batch)
                    labels = moved["labels"]
                    target_tokens = int(labels[:, 1:].ne(-100).sum().detach().cpu())
                    if target_tokens == 0:
                        raise ValueError("batch contains no supervised target tokens")
                    accumulated_target_tokens += target_tokens
                    accumulated_examples += int(labels.shape[0])

                    with self._autocast():
                        output = self._model_forward(moved)
                        loss, components = causal_language_model_loss(
                            output.logits,
                            labels,
                            auxiliary_losses=output.auxiliary_losses,
                        )
                        scaled_loss = loss / self.config.gradient_accumulation_steps

                    self.scaler.scale(scaled_loss).backward()
                    accumulated_total_loss += float(loss.detach().float().cpu())
                    accumulated_language_loss += float(
                        components["language_model"].detach().float().cpu()
                    )
                    accumulated_auxiliary_loss += sum(
                        float(value.detach().float().cpu())
                        for name, value in components.items()
                        if name != "language_model"
                    )
                    progress.micro_step += 1
                    progress.data_cursor += int(labels.shape[0])
        except RuntimeError as exc:
            if self._is_out_of_memory(exc):
                self.optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise TrainingOOMError(self._oom_message()) from exc
            raise

        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_gradient_norm
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        progress.global_step += 1
        progress.consumed_tokens += accumulated_target_tokens
        progress.consumed_examples += accumulated_examples
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        divisor = self.config.gradient_accumulation_steps
        return TrainingMetrics(
            step=progress.global_step,
            loss=accumulated_total_loss / divisor,
            language_model_loss=accumulated_language_loss / divisor,
            auxiliary_loss=accumulated_auxiliary_loss / divisor,
            gradient_norm=float(gradient_norm.detach().float().cpu()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            target_tokens=accumulated_target_tokens,
            consumed_tokens=progress.consumed_tokens,
            examples=accumulated_examples,
            consumed_examples=progress.consumed_examples,
            step_time_ms=elapsed_ms,
            target_tokens_per_second=(
                accumulated_target_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
            ),
        )

    def _move_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"input_ids", "labels"}
        if not required.issubset(batch):
            missing = sorted(required - set(batch))
            raise ValueError(f"batch is missing required tensors: {', '.join(missing)}")
        moved: dict[str, torch.Tensor] = {}
        for name in ("input_ids", "labels", "attention_mask", "position_ids"):
            value = batch.get(name)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"batch field {name!r} must be a tensor")
                moved[name] = value.to(self.device, non_blocking=self.device.type == "cuda")
        if moved["input_ids"].shape != moved["labels"].shape:
            raise ValueError("input_ids and labels must have identical shapes")
        return moved

    def _model_forward(self, batch: Mapping[str, torch.Tensor]) -> ModelOutput:
        kwargs: dict[str, Any] = {"return_dict": True}
        if "attention_mask" in batch:
            kwargs["attention_mask"] = batch["attention_mask"]
        if "position_ids" in batch:
            kwargs["position_ids"] = batch["position_ids"]
        if hasattr(self.model, "recurrent"):
            kwargs["mode"] = self.config.model_mode
            if self.config.recurrent_steps is not None:
                kwargs["recurrent_steps"] = self.config.recurrent_steps
        output = self.model(batch["input_ids"], **kwargs)
        if not isinstance(output, ModelOutput):
            raise TypeError("model must return ModelOutput when return_dict=True")
        return output

    @staticmethod
    def _next_batch(iterator: Iterator[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise RuntimeError("batch iterator was exhausted before the training budget") from exc

    @staticmethod
    def _is_out_of_memory(exc: RuntimeError) -> bool:
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def _oom_message(self) -> str:
        return (
            "training ran out of memory. Reduce batch_size or sequence length, increase "
            "gradient_accumulation_steps, use bf16/fp16 on a supported GPU, or reduce model width. "
            f"device={self.device}"
        )

    def _save_checkpoint(
        self,
        path: Path,
        progress: TrainingProgress,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        overwrite: bool = False,
    ) -> Path:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            progress=progress,
            config=self._resolved_checkpoint_config(),
            metadata={"trainer": "wai_r0.training.engine.Trainer"},
            data_state=self._source_state(batches),
            overwrite=overwrite,
        )

    def _maybe_checkpoint(
        self,
        progress: TrainingProgress,
        batches: Iterable[Mapping[str, torch.Tensor]],
    ) -> None:
        interval = self.config.checkpoint_every
        if not interval or progress.global_step % interval:
            return
        destination = Path(self.config.checkpoint_dir) / f"step-{progress.global_step:08d}.pt"
        self._save_checkpoint(destination, progress, batches)
