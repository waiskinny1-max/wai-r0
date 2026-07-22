from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wai_r0.config import ReasonerConfig
from wai_r0.model import ModelOutput, ReasonerCore
from wai_r0.training.losses import causal_language_model_loss


@dataclass(frozen=True, slots=True)
class CalibrationAttempt:
    batch_size: int
    sequence_length: int
    precision: str
    success: bool
    peak_allocated_bytes: int | None
    step_time_ms: float | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    device: str
    attempts: list[CalibrationAttempt]
    recommended_batch_size: int | None
    recommended_sequence_length: int | None
    recommended_precision: str | None
    target_memory_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_sequence_length": self.recommended_sequence_length,
            "recommended_precision": self.recommended_precision,
            "target_memory_fraction": self.target_memory_fraction,
        }


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _attempt(
    config: ReasonerConfig,
    *,
    batch_size: int,
    sequence_length: int,
    precision: str,
    target_memory_fraction: float,
) -> CalibrationAttempt:
    device = torch.device(config.device)
    if device.type != "cuda":
        return CalibrationAttempt(
            batch_size,
            sequence_length,
            precision,
            False,
            None,
            None,
            "calibration requires a CUDA model configuration",
        )
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = ReasonerCore(config)
        model.train()
        tokens = torch.randint(0, config.vocab_size, (batch_size, sequence_length), device=device)
        labels = tokens.clone()
        started = time.perf_counter_ns()
        with _autocast(device, precision):
            output = model(tokens, return_dict=True)
            if not isinstance(output, ModelOutput):
                raise RuntimeError("model did not return structured output")
            loss, _ = causal_language_model_loss(output.logits, labels)
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        peak = int(torch.cuda.max_memory_allocated(device))
        total = int(torch.cuda.get_device_properties(device).total_memory)
        if peak > int(total * target_memory_fraction):
            return CalibrationAttempt(
                batch_size,
                sequence_length,
                precision,
                False,
                peak,
                elapsed_ms,
                "successful step exceeded the configured VRAM safety fraction",
            )
        return CalibrationAttempt(
            batch_size, sequence_length, precision, True, peak, elapsed_ms, None
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return CalibrationAttempt(
            batch_size,
            sequence_length,
            precision,
            False,
            None,
            None,
            "CUDA out of memory",
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return CalibrationAttempt(
                batch_size,
                sequence_length,
                precision,
                False,
                None,
                None,
                "CUDA out of memory",
            )
        raise


def calibrate_model(
    config: ReasonerConfig,
    *,
    batch_sizes: list[int] | None = None,
    sequence_lengths: list[int] | None = None,
    precisions: list[str] | None = None,
    target_memory_fraction: float = 0.90,
) -> CalibrationResult:
    if not 0.1 <= target_memory_fraction <= 0.98:
        raise ValueError("target_memory_fraction must be in [0.1, 0.98]")
    batches = batch_sizes or [1, 2, 4, 8]
    lengths = sequence_lengths or [64, 128, 256, 512]
    modes = precisions or (
        ["bf16", "fp16", "none"] if config.device.startswith("cuda") else ["none"]
    )
    if any(value < 1 for value in batches + lengths):
        raise ValueError("batch sizes and sequence lengths must be positive")
    if any(mode not in {"none", "fp16", "bf16"} for mode in modes):
        raise ValueError("precision must be none/fp16/bf16")
    if not config.device.startswith("cuda") or not torch.cuda.is_available():
        return CalibrationResult(
            device=config.device,
            attempts=[],
            recommended_batch_size=None,
            recommended_sequence_length=None,
            recommended_precision=None,
            target_memory_fraction=target_memory_fraction,
        )

    attempts: list[CalibrationAttempt] = []
    best: CalibrationAttempt | None = None
    for precision in modes:
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            continue
        for sequence_length in lengths:
            if sequence_length > config.max_seq_len:
                continue
            for batch_size in batches:
                attempt = _attempt(
                    config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    precision=precision,
                    target_memory_fraction=target_memory_fraction,
                )
                attempts.append(attempt)
                if attempt.success and (
                    best is None
                    or batch_size * sequence_length > best.batch_size * best.sequence_length
                ):
                    best = attempt
                if not attempt.success and attempt.error == "CUDA out of memory":
                    break
    return CalibrationResult(
        device=config.device,
        attempts=attempts,
        recommended_batch_size=best.batch_size if best else None,
        recommended_sequence_length=best.sequence_length if best else None,
        recommended_precision=best.precision if best else None,
        target_memory_fraction=target_memory_fraction,
    )


__all__ = ["CalibrationAttempt", "CalibrationResult", "calibrate_model"]
