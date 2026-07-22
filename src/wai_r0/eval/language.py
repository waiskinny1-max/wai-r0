from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wai_r0.model import ModelOutput, ReasonerCore
from wai_r0.training.losses import causal_language_model_loss


@dataclass(frozen=True, slots=True)
class LanguageEvaluation:
    batches: int
    examples: int
    target_tokens: int
    mean_loss: float
    perplexity: float
    bits_per_target_token: float
    bits_per_byte: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_language_batches(
    model: ReasonerCore,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    max_batches: int = 16,
    raw_bytes: int | None = None,
) -> LanguageEvaluation:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    was_training = model.training
    model.eval()
    total_nll = 0.0
    target_tokens = 0
    examples = 0
    batch_count = 0
    try:
        for batch in batches:
            if batch_count >= max_batches:
                break
            input_ids = batch.get("input_ids")
            labels = batch.get("labels")
            if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise ValueError("language batch must contain input_ids and labels tensors")
            moved_ids = input_ids.to(model.device_obj)
            moved_labels = labels.to(model.device_obj)
            kwargs: dict[str, Any] = {"return_dict": True}
            attention_mask = batch.get("attention_mask")
            position_ids = batch.get("position_ids")
            if isinstance(attention_mask, torch.Tensor):
                kwargs["attention_mask"] = attention_mask.to(model.device_obj)
            if isinstance(position_ids, torch.Tensor):
                kwargs["position_ids"] = position_ids.to(model.device_obj)
            output = model(moved_ids, **kwargs)
            if not isinstance(output, ModelOutput):
                raise RuntimeError("language evaluation requires structured model output")
            loss, _ = causal_language_model_loss(output.logits, moved_labels)
            count = int(moved_labels[:, 1:].ne(-100).sum().detach().cpu())
            if count == 0:
                continue
            total_nll += float(loss.detach().float().cpu()) * count
            target_tokens += count
            examples += int(input_ids.shape[0])
            batch_count += 1
    finally:
        model.train(was_training)
    if target_tokens == 0:
        raise ValueError("language evaluation observed no target tokens")
    mean_loss = total_nll / target_tokens
    return LanguageEvaluation(
        batches=batch_count,
        examples=examples,
        target_tokens=target_tokens,
        mean_loss=mean_loss,
        perplexity=math.exp(min(mean_loss, 80.0)),
        bits_per_target_token=mean_loss / math.log(2.0),
        bits_per_byte=(total_nll / (raw_bytes * math.log(2.0)) if raw_bytes else None),
    )


__all__ = ["LanguageEvaluation", "evaluate_language_batches"]
