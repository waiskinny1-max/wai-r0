from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from wai_r0.data.chat import IGNORE_INDEX


def causal_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    auxiliary_losses: Mapping[str, torch.Tensor] | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute next-token cross-entropy plus explicitly named auxiliary losses."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocabulary]")
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, time] matching logits")
    if logits.shape[1] < 2:
        raise ValueError("at least two sequence positions are required")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    if not shifted_labels.ne(ignore_index).any():
        raise ValueError("batch contains no supervised target tokens after causal shift")
    language_loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )
    components = {"language_model": language_loss}
    if auxiliary_losses:
        for name, value in auxiliary_losses.items():
            if value.ndim != 0:
                raise ValueError(f"auxiliary loss {name!r} must be scalar")
            components[name] = value
    total = torch.stack(tuple(components.values())).sum()
    return total, components
