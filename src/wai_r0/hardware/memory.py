from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    cache_bytes: int
    estimated_total_bytes: int
    assumptions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def estimate_training_memory(
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    n_layers: int,
    cache_bytes: int = 0,
    optimizer: str = "adamw",
    mixed_precision: str = "none",
    activation_checkpointing: bool = False,
) -> MemoryEstimate:
    if batch_size < 1 or sequence_length < 1 or d_model < 1 or n_layers < 1:
        raise ValueError("memory-estimate dimensions must be positive")
    params = parameter_bytes(model)
    gradient = params
    if optimizer != "adamw":
        raise ValueError("only AdamW memory estimation is currently supported")
    optimizer_bytes = params * 2
    if mixed_precision in {"fp16", "bf16"}:
        optimizer_bytes += params
    elif mixed_precision != "none":
        raise ValueError("mixed_precision must be none/fp16/bf16")
    element_size = 2 if mixed_precision in {"fp16", "bf16"} else 4
    activation_multiplier = 8 if activation_checkpointing else 18
    activations = (
        batch_size * sequence_length * d_model * n_layers * element_size * activation_multiplier
    )
    total = params + gradient + optimizer_bytes + activations + cache_bytes
    return MemoryEstimate(
        parameter_bytes=params,
        gradient_bytes=gradient,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activations,
        cache_bytes=cache_bytes,
        estimated_total_bytes=total,
        assumptions=[
            "AdamW keeps two moment buffers.",
            "Activation estimate is a conservative architecture-independent approximation.",
            "Allocator fragmentation and framework overhead are not included.",
        ],
    )


def cuda_memory_snapshot(device: str | torch.device = "cuda") -> dict[str, int]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA memory snapshot requires an available CUDA device")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


__all__ = [
    "MemoryEstimate",
    "cuda_memory_snapshot",
    "estimate_training_memory",
    "parameter_bytes",
]
