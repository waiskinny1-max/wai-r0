from __future__ import annotations

import torch

from wai_r0.config import ReasonerConfig
from wai_r0.experiments.statistics import compare_paired, summarize_samples
from wai_r0.model import ReasonerCore
from wai_r0.profiler import profile_model
from wai_r0.training.engine import Trainer, TrainerConfig


def _model() -> ReasonerCore:
    return ReasonerCore(
        ReasonerConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=4,
            d_ff=32,
            max_seq_len=16,
            seed=3,
        )
    )


def test_trainer_runs_and_counts_target_tokens() -> None:
    model = _model()
    config = TrainerConfig(max_steps=2, learning_rate=1e-3)
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, 3, 4]]),
        },
        {
            "input_ids": torch.tensor([[2, 3, 4, 5]]),
            "labels": torch.tensor([[-100, -100, 4, 5]]),
        },
    ]
    result = Trainer(model, config).train(batches)
    assert result.progress.global_step == 2
    assert result.progress.consumed_tokens == 4
    assert len(result.metrics) == 2
    assert all(metric.loss > 0 for metric in result.metrics)


def test_cpu_profiler_reports_measured_cache_and_restores_mode() -> None:
    model = _model().transformer.train()
    profile = profile_model(
        model,
        batch_size=1,
        sequence_length=4,
        warmup_runs=0,
        measured_runs=1,
    )
    assert profile.kv_cache_bytes > 0
    assert profile.prefill_latency_ms_median > 0
    assert profile.decode_latency_ms_median > 0
    assert profile.peak_allocated_bytes is None
    assert model.training is True


def test_descriptive_and_paired_statistics() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0])
    assert summary.count == 3
    assert summary.mean == 2.0
    comparison = compare_paired([0.8, 0.9, 1.0], [0.7, 0.85, 0.95])
    assert comparison.count == 3
    assert comparison.wins == 3
    assert comparison.losses == 0
    assert comparison.mean_difference > 0


def test_temporary_torch_threads_restores_after_success_and_failure() -> None:
    import pytest

    from wai_r0.core.runtime import temporary_torch_threads

    original = torch.get_num_threads()
    requested = 1 if original != 1 else 2
    with temporary_torch_threads(requested) as effective:
        assert effective == requested
        assert torch.get_num_threads() == requested
    assert torch.get_num_threads() == original

    with (
        pytest.raises(RuntimeError, match="boom"),
        temporary_torch_threads(requested),
    ):
        assert torch.get_num_threads() == requested
        raise RuntimeError("boom")
    assert torch.get_num_threads() == original

    with (
        pytest.raises(ValueError, match="positive integer"),
        temporary_torch_threads(0),
    ):
        pass


def test_trainer_cpu_thread_policy_is_scoped_and_reported_in_config() -> None:
    original = torch.get_num_threads()
    requested = 1 if original != 1 else 2
    model = _model()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, learning_rate=1e-3, cpu_threads=requested),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }
    observed: list[int] = []
    trainer.train([batch], event_callback=lambda _metric: observed.append(torch.get_num_threads()))
    assert observed == [requested]
    assert torch.get_num_threads() == original
    assert trainer._resolved_checkpoint_config()["trainer"]["cpu_threads"] == requested


def test_profiler_cpu_thread_policy_is_scoped_and_recorded() -> None:
    original = torch.get_num_threads()
    requested = 1 if original != 1 else 2
    profile = profile_model(
        _model().transformer,
        batch_size=1,
        sequence_length=4,
        warmup_runs=0,
        measured_runs=1,
        cpu_threads=requested,
    )
    assert profile.cpu_threads == requested
    assert torch.get_num_threads() == original
