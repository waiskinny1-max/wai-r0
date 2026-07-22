from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int | None = None

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when set")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.min_p is not None and not 0 <= self.min_p < 1:
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for row in range(previous_tokens.shape[0]):
        unique_tokens = previous_tokens[row].unique()
        selected = adjusted[row, unique_tokens]
        adjusted[row, unique_tokens] = torch.where(
            selected < 0,
            selected * penalty,
            selected / penalty,
        )
    return adjusted


def _top_k_filter(scores: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k >= scores.shape[-1]:
        return scores
    threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
    return scores.masked_fill(scores < threshold, float("-inf"))


def _top_p_filter(scores: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return scores
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    probabilities = sorted_scores.softmax(dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scores, float("-inf"))
    return filtered.scatter(dim=-1, index=sorted_indices, src=sorted_scores)


def _min_p_filter(scores: torch.Tensor, min_p: float | None) -> torch.Tensor:
    if min_p is None or min_p <= 0:
        return scores
    probabilities = scores.softmax(dim=-1)
    threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
    return scores.masked_fill(probabilities < threshold, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    previous_tokens: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    config.validate()
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if previous_tokens.ndim != 2 or previous_tokens.shape[0] != logits.shape[0]:
        raise ValueError("previous_tokens must have shape [batch, time]")
    adjusted = _apply_repetition_penalty(logits.float(), previous_tokens, config.repetition_penalty)
    if not config.do_sample:
        return adjusted.argmax(dim=-1, keepdim=True)
    scores = adjusted / config.temperature
    scores = _top_k_filter(scores, config.top_k)
    scores = _top_p_filter(scores, config.top_p)
    scores = _min_p_filter(scores, config.min_p)
    probabilities = scores.softmax(dim=-1)
    if not bool(torch.isfinite(probabilities).all().detach().cpu()):
        raise FloatingPointError("sampling probabilities are non-finite")
    if bool((probabilities.sum(dim=-1) <= 0).any().detach().cpu()):
        raise FloatingPointError("sampling filters removed every token")
    return torch.multinomial(probabilities, 1, generator=generator)


__all__ = ["SamplingConfig", "sample_next_token"]
