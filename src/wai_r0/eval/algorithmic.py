from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from wai_r0.data.chat import IGNORE_INDEX

AlgorithmicTask = Literal[
    "copy",
    "reverse",
    "parity",
    "modular_addition",
    "sorting",
    "selective_copy",
    "associative_recall",
    "bracket_balance",
    "finite_state_parity",
]

PAD = 0
BOS = 1
SEP = 2
EOS = 3
TASK_BASE = 4
DATA_BASE = 32
DEFAULT_VOCAB_SIZE = 96
_TASK_IDS: dict[AlgorithmicTask, int] = {
    "copy": TASK_BASE,
    "reverse": TASK_BASE + 1,
    "parity": TASK_BASE + 2,
    "modular_addition": TASK_BASE + 3,
    "sorting": TASK_BASE + 4,
    "selective_copy": TASK_BASE + 5,
    "associative_recall": TASK_BASE + 6,
    "bracket_balance": TASK_BASE + 7,
    "finite_state_parity": TASK_BASE + 8,
}


@dataclass(frozen=True, slots=True)
class AlgorithmicExample:
    task: AlgorithmicTask
    prompt: tuple[int, ...]
    answer: tuple[int, ...]
    difficulty: int
    metadata: dict[str, Any]

    @property
    def sequence(self) -> tuple[int, ...]:
        return (BOS, _TASK_IDS[self.task], *self.prompt, SEP, *self.answer, EOS)

    @property
    def first_target_index(self) -> int:
        return 3 + len(self.prompt)

    def validate(self, *, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        if self.difficulty < 1:
            raise ValueError("difficulty must be positive")
        if not self.answer:
            raise ValueError("algorithmic answers cannot be empty")
        if min(self.sequence) < 0 or max(self.sequence) >= vocab_size:
            raise ValueError("algorithmic token is outside the configured vocabulary")


@dataclass(frozen=True, slots=True)
class AlgorithmicEvaluation:
    task: str
    length: int
    examples: int
    exact_match: float
    token_accuracy: float
    mean_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _data(value: int) -> int:
    return DATA_BASE + value


def _random_values(rng: random.Random, length: int, *, cardinality: int = 16) -> list[int]:
    return [rng.randrange(cardinality) for _ in range(length)]


def generate_algorithmic_example(
    task: AlgorithmicTask,
    *,
    length: int,
    rng: random.Random,
) -> AlgorithmicExample:
    if task not in _TASK_IDS:
        raise ValueError(f"unsupported algorithmic task: {task}")
    if length < 1:
        raise ValueError("length must be positive")

    metadata: dict[str, Any] = {}
    if task == "copy":
        values = _random_values(rng, length)
        prompt = tuple(_data(value) for value in values)
        answer = prompt
    elif task == "reverse":
        values = _random_values(rng, length)
        prompt = tuple(_data(value) for value in values)
        answer = tuple(reversed(prompt))
    elif task == "parity":
        bits = [rng.randrange(2) for _ in range(length)]
        prompt = tuple(_data(bit) for bit in bits)
        answer = (_data(sum(bits) % 2),)
    elif task == "modular_addition":
        modulus = max(2, min(16, length + 2))
        left = rng.randrange(modulus)
        right = rng.randrange(modulus)
        prompt = (_data(left), _data(right), _data(modulus))
        answer = (_data((left + right) % modulus),)
        metadata = {"modulus": modulus}
    elif task == "sorting":
        values = _random_values(rng, length, cardinality=min(16, max(2, length)))
        prompt = tuple(_data(value) for value in values)
        answer = tuple(_data(value) for value in sorted(values))
    elif task == "selective_copy":
        values = _random_values(rng, length)
        selectors = [rng.randrange(2) for _ in range(length)]
        if not any(selectors):
            selectors[rng.randrange(length)] = 1
        encoded: list[int] = []
        selected: list[int] = []
        for selector, value in zip(selectors, values, strict=True):
            encoded.extend((_data(16 + selector), _data(value)))
            if selector:
                selected.append(_data(value))
        prompt = tuple(encoded)
        answer = tuple(selected)
    elif task == "associative_recall":
        pair_count = max(2, length)
        keys = rng.sample(range(16), k=min(pair_count, 16))
        values = _random_values(rng, len(keys))
        query_index = rng.randrange(len(keys))
        encoded = []
        for key, value in zip(keys, values, strict=True):
            encoded.extend((_data(key), _data(16 + value)))
        encoded.extend((SEP + 16, _data(keys[query_index])))
        prompt = tuple(encoded)
        answer = (_data(16 + values[query_index]),)
        metadata = {"pairs": len(keys)}
    elif task == "bracket_balance":
        # Half the examples are generated from balanced pairs, the rest receive
        # a deterministic corruption. This exercises stack-like state without
        # relying on natural-language templates.
        balanced = rng.randrange(2) == 0
        pairs = max(1, length // 2)
        tokens: list[int] = []
        depth = 0
        for index in range(pairs * 2):
            can_close = depth > 0
            remaining = pairs * 2 - index
            must_close = depth == remaining
            open_next = not can_close or (not must_close and rng.randrange(2) == 0)
            if open_next:
                tokens.append(_data(24))
                depth += 1
            else:
                tokens.append(_data(25))
                depth -= 1
        tokens.extend([_data(25)] * depth)
        tokens = tokens[: pairs * 2]
        if not balanced:
            corrupt = rng.randrange(len(tokens))
            tokens[corrupt] = _data(25) if tokens[corrupt] == _data(24) else _data(24)
        prompt = tuple(tokens)
        answer = (_data(1 if _is_balanced(tokens) else 0),)
        metadata = {"requested_balanced": balanced}
    else:  # finite_state_parity
        symbols = _random_values(rng, length, cardinality=4)
        state = 0
        for symbol in symbols:
            state = (state + (symbol % 2)) % 2
        prompt = tuple(_data(symbol) for symbol in symbols)
        answer = (_data(state),)

    example = AlgorithmicExample(task, prompt, answer, length, metadata)
    example.validate()
    return example


def _is_balanced(tokens: Sequence[int]) -> bool:
    depth = 0
    for token in tokens:
        if token == _data(24):
            depth += 1
        elif token == _data(25):
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def encode_algorithmic_example(
    example: AlgorithmicExample,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    example.validate()
    sequence = list(example.sequence)
    if len(sequence) > max_length:
        raise ValueError(
            f"encoded {example.task} example requires {len(sequence)} positions, "
            f"but max_length is {max_length}"
        )
    labels = [IGNORE_INDEX] * example.first_target_index + sequence[example.first_target_index :]
    return torch.tensor(sequence, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_algorithmic(
    examples: Sequence[AlgorithmicExample],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples cannot be empty")
    encoded = [encode_algorithmic_example(example, max_length=max_length) for example in examples]
    width = max(tokens.numel() for tokens, _ in encoded)
    input_ids = torch.full((len(encoded), width), PAD, dtype=torch.long)
    labels = torch.full((len(encoded), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row, (tokens, target) in enumerate(encoded):
        input_ids[row, : tokens.numel()] = tokens
        labels[row, : target.numel()] = target
        attention_mask[row, : tokens.numel()] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass(slots=True)
class AlgorithmicStreamState:
    batches_emitted: int = 0
    examples_emitted: int = 0


class AlgorithmicBatchStream(Iterator[Mapping[str, torch.Tensor]]):
    """Deterministic generated source whose RNG and cursor are checkpointable."""

    def __init__(
        self,
        task: AlgorithmicTask,
        *,
        seed: int,
        batch_size: int,
        lengths: Sequence[int],
        max_length: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError("lengths must contain positive values")
        self.task = task
        self.seed = seed
        self.batch_size = batch_size
        self.lengths = tuple(int(length) for length in lengths)
        self.max_length = max_length
        self._rng = random.Random(seed)
        self.state = AlgorithmicStreamState()
        # Fail before training instead of after consuming a partial run.
        probe_rng = random.Random(seed)
        for length in self.lengths:
            encode_algorithmic_example(
                generate_algorithmic_example(task, length=length, rng=probe_rng),
                max_length=max_length,
            )

    def __iter__(self) -> AlgorithmicBatchStream:
        return self

    def __next__(self) -> Mapping[str, torch.Tensor]:
        examples = [
            generate_algorithmic_example(
                self.task,
                length=self.lengths[self._rng.randrange(len(self.lengths))],
                rng=self._rng,
            )
            for _ in range(self.batch_size)
        ]
        self.state.batches_emitted += 1
        self.state.examples_emitted += len(examples)
        return collate_algorithmic(examples, max_length=self.max_length)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "task": self.task,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "lengths": list(self.lengths),
            "max_length": self.max_length,
            "state": asdict(self.state),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        expected = {
            "format_version": 1,
            "task": self.task,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "lengths": list(self.lengths),
            "max_length": self.max_length,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"algorithmic stream {name} does not match checkpoint")
        state = payload.get("state")
        rng_state = payload.get("rng_state")
        if not isinstance(state, Mapping) or rng_state is None:
            raise ValueError("algorithmic stream checkpoint is incomplete")
        self.state = AlgorithmicStreamState(**dict(state))
        self._rng.setstate(_nested_tuple(rng_state))


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def fixed_algorithmic_examples(
    task: AlgorithmicTask,
    *,
    seed: int,
    length: int,
    count: int,
) -> list[AlgorithmicExample]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return [generate_algorithmic_example(task, length=length, rng=rng) for _ in range(count)]


__all__ = [
    "DEFAULT_VOCAB_SIZE",
    "AlgorithmicBatchStream",
    "AlgorithmicEvaluation",
    "AlgorithmicExample",
    "AlgorithmicTask",
    "collate_algorithmic",
    "encode_algorithmic_example",
    "fixed_algorithmic_examples",
    "generate_algorithmic_example",
]
