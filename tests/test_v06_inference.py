from __future__ import annotations

import torch

from wai_r0.config import ReasonerConfig
from wai_r0.eval.generation import diagnose_generation
from wai_r0.inference import SamplingConfig, generate_tokens, sample_next_token
from wai_r0.model import ReasonerCore


def _model() -> ReasonerCore:
    return ReasonerCore(
        ReasonerConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=4,
            d_ff=32,
            max_seq_len=24,
            seed=7,
        )
    )


def test_sampling_is_seeded_and_filters() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    previous = torch.tensor([[0, 1]])
    config = SamplingConfig(do_sample=True, top_p=0.8, min_p=0.05, seed=5)
    first_generator = torch.Generator().manual_seed(5)
    second_generator = torch.Generator().manual_seed(5)
    first = sample_next_token(
        logits, previous_tokens=previous, config=config, generator=first_generator
    )
    second = sample_next_token(
        logits, previous_tokens=previous, config=config, generator=second_generator
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_cached_and_uncached_native_generation_match_greedy() -> None:
    model = _model()
    prompt = torch.tensor([[1, 2, 3]])
    cached = generate_tokens(model, prompt, max_new_tokens=4, use_cache=True)
    uncached = generate_tokens(model, prompt, max_new_tokens=4, use_cache=False)
    torch.testing.assert_close(cached.token_ids, uncached.token_ids, rtol=0, atol=0)
    assert cached.generated_tokens == 4


def test_generation_diagnostics_measure_repetition() -> None:
    tokens = torch.tensor([[1, 2, 3, 3, 3, 4], [1, 2, 5, 6, 7, 8]])
    diagnostics = diagnose_generation(tokens, prompt_length=2)
    assert diagnostics.generated_tokens == 8
    assert diagnostics.longest_repeated_run == 3
    assert diagnostics.adjacent_repetition_fraction > 0
