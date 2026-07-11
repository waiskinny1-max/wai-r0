from __future__ import annotations

import os
import random
from collections.abc import Generator, Iterable
from contextlib import contextmanager

import torch
from torch import nn


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python and Torch without silently changing the determinism policy."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@contextmanager
def temporary_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily seed Python/Torch and restore all RNG streams afterwards."""

    from wai_r0.core.reproducibility import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def scalar_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def tensor_to_float_list(value: torch.Tensor | Iterable[float]) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().float().cpu().tolist()]
    return [float(item) for item in value]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 1 or eps <= 0:
            raise ValueError("RMSNorm dim and eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Cached RoPE values with arbitrary batched position IDs."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        if max_seq_len < 1 or base <= 1:
            raise ValueError("invalid RoPE configuration")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.inv_freq: torch.Tensor
        self._cos_cached: torch.Tensor
        self._sin_cached: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
        self._cached_length = 0
        self._cached_device: torch.device | None = None

    def _ensure_cache(self, length: int, *, device: torch.device) -> None:
        if length > self.max_seq_len:
            raise ValueError("position exceeds configured max_seq_len")
        if (
            length <= self._cached_length
            and self._cached_device == device
            and self._cos_cached.device == device
        ):
            return
        positions = torch.arange(length, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device)
        phase = torch.outer(positions, inv_freq)
        self._cos_cached = phase.cos()
        self._sin_cached = phase.sin()
        self._cached_length = length
        self._cached_device = device

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, time]")
        if position_ids.numel() == 0:
            raise ValueError("position_ids cannot be empty")
        if position_ids.dtype not in (torch.int32, torch.int64):
            position_ids = position_ids.long()
        minimum = int(position_ids.min().detach().cpu())
        maximum = int(position_ids.max().detach().cpu())
        if minimum < 0:
            raise ValueError("position_ids cannot be negative")
        self._ensure_cache(maximum + 1, device=position_ids.device)
        flat = position_ids.reshape(-1)
        cos = self._cos_cached.index_select(0, flat).view(*position_ids.shape, -1)
        sin = self._sin_cached.index_select(0, flat).view(*position_ids.shape, -1)
        return cos[:, None].to(dtype=dtype), sin[:, None].to(dtype=dtype)


def rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    offset: int = 0,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility helper for v0.4 callers using contiguous positions."""

    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if offset < 0:
        raise ValueError("offset cannot be negative")
    frequencies = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    phase = torch.outer(positions, frequencies)
    return phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("RoPE input dimension must be even")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin caches must have identical shapes")
    if cos.ndim == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    elif cos.ndim == 3:
        cos = cos[:, None]
        sin = sin[:, None]
    if cos.ndim != 4:
        raise ValueError("RoPE cache must be [T,D/2], [B,T,D/2], or [B,1,T,D/2]")
    if cos.shape[-2:] != (x.shape[-2], x.shape[-1] // 2):
        raise ValueError("RoPE cache shape does not match the input")
    if cos.shape[0] not in (1, x.shape[0]):
        raise ValueError("RoPE cache batch dimension does not match the input")

    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if x.ndim != 4:
        raise ValueError("KV tensor must have shape [batch, heads, time, dim]")
    if repeats == 1:
        return x
    batch, heads, time, dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, repeats, time, dim)
        .reshape(batch, heads * repeats, time, dim)
    )
