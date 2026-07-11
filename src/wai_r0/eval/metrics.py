from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from wai_r0.data.chat import IGNORE_INDEX
from wai_r0.model import ModelOutput


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    examples: int
    exact_matches: int
    correct_tokens: int
    target_tokens: int
    loss_sum: float

    @property
    def exact_match(self) -> float:
        return self.exact_matches / self.examples if self.examples else 0.0

    @property
    def token_accuracy(self) -> float:
        return self.correct_tokens / self.target_tokens if self.target_tokens else 0.0

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.target_tokens if self.target_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "exact_match": self.exact_match,
            "token_accuracy": self.token_accuracy,
            "mean_loss": self.mean_loss,
            "correct_tokens": self.correct_tokens,
            "target_tokens": self.target_tokens,
        }


@torch.inference_mode()
def evaluate_sequence_batches(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    max_batches: int,
    model_mode: str = "fast",
    recurrent_steps: int | None = None,
) -> SequenceMetrics:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if model_mode not in {"fast", "think"}:
        raise ValueError("model_mode must be fast or think")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    examples = exact_matches = correct_tokens = target_tokens = 0
    loss_sum = 0.0
    try:
        for index, batch in enumerate(batches):
            if index >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            kwargs: dict[str, Any] = {
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if hasattr(model, "recurrent"):
                kwargs["mode"] = model_mode
                if recurrent_steps is not None:
                    kwargs["recurrent_steps"] = recurrent_steps
            output = model(input_ids, **kwargs)
            if not isinstance(output, ModelOutput):
                raise TypeError("model must return ModelOutput")
            predictions = output.logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            supervised = targets.ne(IGNORE_INDEX)
            per_token_loss = F.cross_entropy(
                output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(targets)
            matches = predictions.eq(targets) | ~supervised
            examples += int(targets.shape[0])
            exact_matches += int(matches.all(dim=1).sum().cpu())
            correct_tokens += int((predictions.eq(targets) & supervised).sum().cpu())
            target_tokens += int(supervised.sum().cpu())
            loss_sum += float(per_token_loss[supervised].sum().cpu())
    finally:
        model.train(was_training)
    if examples == 0 or target_tokens == 0:
        raise RuntimeError("evaluation produced no supervised examples")
    return SequenceMetrics(examples, exact_matches, correct_tokens, target_tokens, loss_sum)


__all__ = ["SequenceMetrics", "evaluate_sequence_batches"]
