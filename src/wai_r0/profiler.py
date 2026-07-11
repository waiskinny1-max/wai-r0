from __future__ import annotations

import platform
import resource
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import cast

import torch

from wai_r0.core.runtime import temporary_torch_threads
from wai_r0.model import DecoderOnlyTransformer, ModelOutput


@dataclass(frozen=True, slots=True)
class ProfileResult:
    device: str
    device_name: str
    dtype: str
    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    kv_cache_bytes: int
    kv_cache_payload_bytes: int
    kv_cache_metadata_bytes: int
    theoretical_kv_cache_bytes: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    process_peak_rss_bytes: int | None
    prefill_latency_ms_median: float
    prefill_latency_ms_p95: float
    decode_latency_ms_median: float
    decode_latency_ms_p95: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    warmup_runs: int
    measured_runs: int
    attention_type: str
    cache_semantics: str
    cpu_threads: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(action: Callable[[], None], *, runs: int, device: torch.device) -> list[float]:
    durations: list[float] = []
    for _ in range(runs):
        _synchronize(device)
        started = time.perf_counter_ns()
        action()
        _synchronize(device)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return durations


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _peak_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ValueError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.
    return value if platform.system() == "Darwin" else value * 1024


def _theoretical_cache_bytes(
    model: DecoderOnlyTransformer,
    *,
    batch_size: int,
    sequence_length: int,
    element_size: int,
) -> int:
    cfg = model.cfg
    head_dim = cfg.d_model // cfg.n_heads
    if cfg.attention_type == "mla_lite":
        per_layer = batch_size * sequence_length * cfg.mla_latent_dim * element_size
    else:
        per_layer = 2 * batch_size * sequence_length * cfg.n_kv_heads * head_dim * element_size
    return per_layer * cfg.n_layers


def profile_model(
    model: DecoderOnlyTransformer,
    *,
    batch_size: int = 1,
    sequence_length: int = 32,
    warmup_runs: int = 2,
    measured_runs: int = 5,
    cpu_threads: int | None = None,
) -> ProfileResult:
    """Measure prefill/decode and cache use under an explicit CPU thread policy."""

    device = next(model.parameters()).device
    requested_threads = cpu_threads if device.type == "cpu" else None
    with temporary_torch_threads(requested_threads) as effective_threads:
        return cast(
            ProfileResult,
            _profile_model_impl(
                model,
                batch_size=batch_size,
                sequence_length=sequence_length,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                cpu_threads=effective_threads if device.type == "cpu" else None,
            ),
        )


@torch.inference_mode()
def _profile_model_impl(
    model: DecoderOnlyTransformer,
    *,
    batch_size: int,
    sequence_length: int,
    warmup_runs: int,
    measured_runs: int,
    cpu_threads: int | None,
) -> ProfileResult:
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    if sequence_length + 1 > model.cfg.max_seq_len:
        raise ValueError("sequence_length + one decode token exceeds max_seq_len")

    parameter = next(model.parameters())
    device = parameter.device
    tokens = torch.randint(
        0,
        model.cfg.vocab_size,
        (batch_size, sequence_length),
        device=device,
    )

    was_training = model.training
    model.eval()
    try:
        for _ in range(warmup_runs):
            model(tokens, use_cache=True, return_dict=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def prefill() -> None:
            model(tokens, use_cache=True, return_dict=True)

        prefill_ms = _measure(prefill, runs=measured_runs, device=device)
        prefill_output = model(tokens, use_cache=True, return_dict=True)
        if not isinstance(prefill_output, ModelOutput) or prefill_output.past_key_values is None:
            raise RuntimeError("model did not return a KV cache")
        cache = prefill_output.past_key_values
        next_token = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        def decode() -> None:
            model(
                next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

        decode_ms = _measure(decode, runs=measured_runs, device=device)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        )
        parameter_count = sum(item.numel() for item in model.parameters())
        trainable_parameter_count = sum(
            item.numel() for item in model.parameters() if item.requires_grad
        )
        parameter_bytes = sum(item.numel() * item.element_size() for item in model.parameters())
        payload_bytes = sum(item.payload_bytes for item in cache)
        metadata_bytes = sum(item.metadata_bytes for item in cache)
        allocated_bytes = sum(item.allocated_bytes for item in cache)
        prefill_median = statistics.median(prefill_ms)
        decode_median = statistics.median(decode_ms)
        cache_semantics = (
            "compressed latent payload; keys/values reconstructed during decode"
            if model.cfg.attention_type == "mla_lite"
            else "materialized key/value tensors"
        )
        return ProfileResult(
            device=str(device),
            device_name=(
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
            dtype=str(parameter.dtype).removeprefix("torch."),
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_parameter_count,
            parameter_bytes=parameter_bytes,
            kv_cache_bytes=allocated_bytes,
            kv_cache_payload_bytes=payload_bytes,
            kv_cache_metadata_bytes=metadata_bytes,
            theoretical_kv_cache_bytes=_theoretical_cache_bytes(
                model,
                batch_size=batch_size,
                sequence_length=sequence_length,
                element_size=parameter.element_size(),
            ),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            process_peak_rss_bytes=_peak_rss_bytes(),
            prefill_latency_ms_median=prefill_median,
            prefill_latency_ms_p95=_percentile(prefill_ms, 0.95),
            decode_latency_ms_median=decode_median,
            decode_latency_ms_p95=_percentile(decode_ms, 0.95),
            prefill_tokens_per_second=(
                (batch_size * sequence_length) / (prefill_median / 1000)
                if prefill_median > 0
                else 0.0
            ),
            decode_tokens_per_second=(
                batch_size / (decode_median / 1000) if decode_median > 0 else 0.0
            ),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            attention_type=model.cfg.attention_type,
            cache_semantics=cache_semantics,
            cpu_threads=cpu_threads,
        )
    finally:
        model.train(was_training)


__all__ = ["ProfileResult", "profile_model"]
