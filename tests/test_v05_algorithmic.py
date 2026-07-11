from __future__ import annotations

import random

import pytest
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.eval.algorithmic import (
    DEFAULT_VOCAB_SIZE,
    AlgorithmicBatchStream,
    collate_algorithmic,
    encode_algorithmic_example,
    fixed_algorithmic_examples,
    generate_algorithmic_example,
)
from wai_r0.eval.metrics import evaluate_sequence_batches
from wai_r0.model import ReasonerCore
from wai_r0.training.engine import Trainer, TrainerConfig

TASKS = [
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


@pytest.mark.parametrize("task", TASKS)
def test_algorithmic_tasks_are_deterministic_and_supervised(task: str) -> None:
    first = generate_algorithmic_example(task, length=5, rng=random.Random(17))
    second = generate_algorithmic_example(task, length=5, rng=random.Random(17))
    assert first == second
    tokens, labels = encode_algorithmic_example(first, max_length=64)
    assert tokens.shape == labels.shape
    assert labels.ne(-100).sum() == len(first.answer) + 1
    assert tokens.max() < DEFAULT_VOCAB_SIZE


def test_algorithmic_stream_restores_exact_next_batch() -> None:
    stream = AlgorithmicBatchStream("reverse", seed=7, batch_size=3, lengths=[3, 5], max_length=32)
    next(stream)
    state = stream.state_dict()
    expected = next(stream)

    restored = AlgorithmicBatchStream(
        "reverse", seed=7, batch_size=3, lengths=[3, 5], max_length=32
    )
    restored.load_state_dict(state)
    actual = next(restored)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_tiny_algorithmic_training_and_evaluation() -> None:
    config = ReasonerConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        d_ff=32,
        max_seq_len=32,
        seed=4,
    )
    model = ReasonerCore(config)
    source = AlgorithmicBatchStream("parity", seed=100, batch_size=4, lengths=[4], max_length=32)
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=2, learning_rate=1e-3, save_final_checkpoint=False),
    )
    result = trainer.train(source)
    assert result.progress.global_step == 2
    metrics = evaluate_sequence_batches(
        model,
        AlgorithmicBatchStream("parity", seed=200, batch_size=4, lengths=[4], max_length=32),
        max_batches=2,
    )
    assert metrics.examples == 8
    assert 0 <= metrics.exact_match <= 1
    assert 0 <= metrics.token_accuracy <= 1
    assert metrics.mean_loss > 0


def test_collate_rejects_too_short_max_length() -> None:
    examples = fixed_algorithmic_examples("copy", seed=1, length=8, count=2)
    with pytest.raises(ValueError, match="requires"):
        collate_algorithmic(examples, max_length=8)
