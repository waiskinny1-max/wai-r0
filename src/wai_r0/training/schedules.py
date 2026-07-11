from __future__ import annotations

import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

ScheduleName = Literal["constant", "linear", "cosine"]


def learning_rate_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: ScheduleName,
    minimum_ratio: float,
) -> float:
    if step < 0:
        raise ValueError("step cannot be negative")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("minimum_ratio must be in [0, 1]")
    if warmup_steps and step < warmup_steps:
        return max(1e-12, (step + 1) / warmup_steps)
    if schedule == "constant":
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    if schedule == "linear":
        decay = 1.0 - progress
    elif schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unsupported schedule: {schedule}")
    return minimum_ratio + (1.0 - minimum_ratio) * decay


def build_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: ScheduleName,
    minimum_ratio: float,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            schedule=schedule,
            minimum_ratio=minimum_ratio,
        ),
    )
