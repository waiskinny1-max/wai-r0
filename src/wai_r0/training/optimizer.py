from __future__ import annotations

from torch import nn
from torch.optim import AdamW, Optimizer


def parameter_groups_for_weight_decay(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    decay = []
    no_decay = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        normalized = name.casefold()
        if (
            parameter.ndim < 2
            or normalized.endswith((".bias", "embed.weight"))
            or "norm" in normalized
            or "embedding" in normalized
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    epsilon: float,
    fused: bool = False,
) -> Optimizer:
    if learning_rate <= 0 or epsilon <= 0:
        raise ValueError("learning_rate and epsilon must be positive")
    if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
        raise ValueError("AdamW betas must be in [0, 1)")
    if fused:
        first_parameter = next(model.parameters(), None)
        if first_parameter is None or first_parameter.device.type != "cuda":
            raise ValueError("fused AdamW requires model parameters on CUDA")
    kwargs: dict[str, object] = {
        "lr": learning_rate,
        "betas": betas,
        "eps": epsilon,
    }
    if fused:
        kwargs["fused"] = True
    try:
        return AdamW(
            parameter_groups_for_weight_decay(model, weight_decay=weight_decay),
            **kwargs,
        )
    except TypeError as exc:
        if fused:
            raise RuntimeError("this PyTorch build does not support fused AdamW") from exc
        raise
