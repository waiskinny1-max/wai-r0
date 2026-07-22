from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from wai_r0.model import ModelOutput, ReasonerCore

ContextTask = Literal["needle", "induction"]


@dataclass(frozen=True, slots=True)
class ContextEvaluation:
    task: ContextTask
    cases: int
    context_length: int
    distractors: int
    exact_accuracy: float
    chance_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _needle_cases(
    *,
    cases: int,
    distractors: int,
    vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_size < 32:
        raise ValueError("context evaluation requires vocab_size >= 32")
    rng = random.Random(seed)
    sequences: list[list[int]] = []
    targets: list[int] = []
    pair_marker, query_marker = 1, 2
    key_low, key_high = 4, min(vocab_size // 2, 64)
    value_low, value_high = key_high, min(vocab_size, key_high + 64)
    if value_high - value_low < 2:
        raise ValueError("vocabulary is too small for disjoint key/value ranges")
    for _ in range(cases):
        keys = rng.sample(range(key_low, key_high), distractors + 1)
        values = [rng.randrange(value_low, value_high) for _ in keys]
        target_index = rng.randrange(len(keys))
        sequence: list[int] = []
        for key, value in zip(keys, values, strict=True):
            sequence.extend((pair_marker, key, value))
        sequence.extend((query_marker, keys[target_index]))
        sequences.append(sequence)
        targets.append(values[target_index])
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _induction_cases(
    *,
    cases: int,
    distractors: int,
    vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_size < 24:
        raise ValueError("induction evaluation requires vocab_size >= 24")
    rng = random.Random(seed)
    sequences: list[list[int]] = []
    targets: list[int] = []
    separator, query = 1, 2
    symbol_low, symbol_high = 4, min(vocab_size, 96)
    for _ in range(cases):
        left = rng.sample(range(symbol_low, symbol_high), distractors + 1)
        right = rng.sample(range(symbol_low, symbol_high), distractors + 1)
        selected = rng.randrange(len(left))
        sequence: list[int] = []
        for source, target in zip(left, right, strict=True):
            sequence.extend((source, separator, target))
        sequence.extend((query, left[selected], separator))
        sequences.append(sequence)
        targets.append(right[selected])
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def evaluate_context_task(
    model: ReasonerCore,
    *,
    task: ContextTask,
    cases: int = 64,
    distractors: int = 4,
    seed: int = 1337,
) -> ContextEvaluation:
    if cases < 1 or distractors < 0:
        raise ValueError("cases must be positive and distractors non-negative")
    if task == "needle":
        inputs, targets = _needle_cases(
            cases=cases,
            distractors=distractors,
            vocab_size=model.cfg.vocab_size,
            seed=seed,
        )
    elif task == "induction":
        inputs, targets = _induction_cases(
            cases=cases,
            distractors=distractors,
            vocab_size=model.cfg.vocab_size,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported context task: {task}")
    if inputs.shape[1] > model.cfg.max_seq_len:
        raise ValueError("generated context exceeds model max_seq_len")
    output = model(inputs.to(model.device_obj), return_dict=True)
    if not isinstance(output, ModelOutput):
        raise RuntimeError("context evaluation requires structured model output")
    predictions = output.logits[:, -1, :].argmax(dim=-1).cpu()
    accuracy = float(predictions.eq(targets).float().mean())
    return ContextEvaluation(
        task=task,
        cases=cases,
        context_length=int(inputs.shape[1]),
        distractors=distractors,
        exact_accuracy=accuracy,
        chance_accuracy=1.0 / model.cfg.vocab_size,
    )


__all__ = ["ContextEvaluation", "ContextTask", "evaluate_context_task"]
