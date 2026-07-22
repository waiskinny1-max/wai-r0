from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class GenerationDiagnostics:
    sequences: int
    generated_tokens: int
    unique_token_fraction: float
    adjacent_repetition_fraction: float
    longest_repeated_run: int
    eos_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_generation(
    token_ids: torch.Tensor,
    *,
    prompt_length: int,
    eos_token_id: int | None = None,
) -> GenerationDiagnostics:
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, time]")
    if prompt_length < 0 or prompt_length > token_ids.shape[1]:
        raise ValueError("invalid prompt_length")
    generated = token_ids[:, prompt_length:]
    count = int(generated.numel())
    if count == 0:
        return GenerationDiagnostics(
            sequences=int(token_ids.shape[0]),
            generated_tokens=0,
            unique_token_fraction=0.0,
            adjacent_repetition_fraction=0.0,
            longest_repeated_run=0,
            eos_fraction=None if eos_token_id is None else 0.0,
        )
    unique_fraction = sum(int(row.unique().numel()) for row in generated) / count
    adjacent = int(generated[:, 1:].eq(generated[:, :-1]).sum()) if generated.shape[1] > 1 else 0
    adjacent_denominator = max(1, generated.shape[0] * max(0, generated.shape[1] - 1))
    longest = 1
    for row in generated.tolist():
        current = 1
        for left, right in pairwise(row):
            if left == right:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
    eos_fraction = None
    if eos_token_id is not None:
        eos_fraction = float(generated.eq(eos_token_id).any(dim=1).float().mean())
    return GenerationDiagnostics(
        sequences=int(token_ids.shape[0]),
        generated_tokens=count,
        unique_token_fraction=float(unique_fraction),
        adjacent_repetition_fraction=adjacent / adjacent_denominator,
        longest_repeated_run=longest,
        eos_fraction=eos_fraction,
    )


__all__ = ["GenerationDiagnostics", "diagnose_generation"]
