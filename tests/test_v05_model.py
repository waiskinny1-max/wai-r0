from __future__ import annotations

import copy

import pytest
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.model import (
    CausalSelfAttention,
    MLALiteAttention,
    ModelOutput,
    ReasonerCore,
    RecurrentRefinement,
    TopKMoE,
)


def make_config(**overrides: object) -> ReasonerConfig:
    values: dict[str, object] = {
        "vocab_size": 41,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_ff": 32,
        "max_seq_len": 24,
        "dropout": 0.0,
        "seed": 7,
    }
    values.update(overrides)
    return ReasonerConfig.from_dict(values)


def test_existing_tensor_api_and_generate_shape() -> None:
    core = ReasonerCore(make_config())
    tokens = torch.randint(0, core.cfg.vocab_size, (2, 8))
    logits = core(tokens)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 8, core.cfg.vocab_size)
    generated = core.transformer.generate(tokens[:1, :2], 2)
    assert generated.shape == (1, 4)


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads", "mla_latent_dim"),
    [("mha", 4, 12), ("gqa", 2, 12), ("mla_lite", 2, 4)],
)
def test_cached_logits_match_full_context(
    attention_type: str,
    n_kv_heads: int,
    mla_latent_dim: int,
) -> None:
    config = make_config(
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        mla_latent_dim=mla_latent_dim,
    )
    model = ReasonerCore(config).transformer.eval()
    prompt = torch.randint(0, config.vocab_size, (2, 5))
    next_token = torch.randint(0, config.vocab_size, (2, 1))

    with torch.inference_mode():
        prefill = model(prompt, use_cache=True, return_dict=True)
        assert isinstance(prefill, ModelOutput)
        assert prefill.past_key_values is not None
        cached = model(
            next_token,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        full = model(torch.cat((prompt, next_token), dim=1), return_dict=True)

    assert isinstance(cached, ModelOutput)
    assert isinstance(full, ModelOutput)
    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-5)
    assert all(item.sequence_length == 6 for item in cached.past_key_values or ())


@pytest.mark.parametrize("attention_type", ["mha", "mla_lite"])
def test_cached_and_uncached_greedy_generation_match(attention_type: str) -> None:
    kwargs: dict[str, object] = {"attention_type": attention_type}
    if attention_type == "mla_lite":
        kwargs.update(n_kv_heads=2, mla_latent_dim=4)
    config = make_config(**kwargs)
    model = ReasonerCore(config).transformer
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    cached = model.generate(prompt, 4, use_cache=True)
    uncached = model.generate(prompt, 4, use_cache=False)
    assert torch.equal(cached, uncached)


def test_generation_restores_training_mode() -> None:
    model = ReasonerCore(make_config(dropout=0.1)).transformer.train()
    model.generate(torch.tensor([[1, 2]]), 1)
    assert model.training is True


def test_effective_dtype_is_applied() -> None:
    core = ReasonerCore(make_config(dtype="bfloat16"))
    assert next(core.parameters()).dtype == torch.bfloat16


def test_cpu_float16_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="float16"):
        ReasonerCore(make_config(dtype="float16", device="cpu"))


def test_state_cache_has_operational_semantics() -> None:
    core = ReasonerCore(make_config())
    state = core.init_state(batch_size=1)
    result = core(torch.tensor([[1, 2, 3]]), state=state, use_cache=True, return_dict=True)
    assert isinstance(result, ModelOutput)
    assert state["past_key_values"] is not None
    second = core(torch.tensor([[4]]), state=state, use_cache=True, return_dict=True)
    assert isinstance(second, ModelOutput)
    assert all(item.sequence_length == 4 for item in second.past_key_values or ())


def test_think_does_not_mutate_frozen_config() -> None:
    config = make_config(recurrent_depth=2)
    core = ReasonerCore(config)
    before = copy.deepcopy(config)
    output = core.think(torch.tensor([[1, 2, 3]]), budget=4)
    assert output.shape == (1, 3, config.vocab_size)
    assert core.cfg == before
    assert core.recurrent is not None
    assert core.recurrent.last_stats is not None
    assert core.recurrent.last_stats.depth == 4


def test_attention_stats_compatibility() -> None:
    config = make_config(n_layers=1)
    attention = CausalSelfAttention(config)
    attention(torch.randn(2, 5, config.d_model))
    assert attention.last_stats is not None
    assert attention.last_stats.kv_cache_bytes > 0

    mla_config = make_config(
        n_layers=1,
        attention_type="mla_lite",
        n_kv_heads=2,
        mla_latent_dim=4,
    )
    mla = MLALiteAttention(mla_config)
    mla(torch.randn(2, 5, mla_config.d_model))
    assert mla.last_stats is not None
    assert 0 < (mla.last_stats.compression_ratio or 0) < 1


def test_moe_exposes_losses_capacity_and_load() -> None:
    config = make_config(
        n_layers=1,
        use_moe=True,
        n_experts=4,
        experts_per_token=2,
        moe_capacity_factor=0.5,
    )
    moe = TopKMoE(config)
    output, auxiliary = moe(torch.randn(2, 8, config.d_model), return_aux=True)
    assert output.shape == (2, 8, config.d_model)
    assert set(auxiliary) == {"moe_load_balance", "moe_router_z"}
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in auxiliary.values())
    assert moe.last_stats is not None
    assert sum(moe.last_stats.load_fraction) == pytest.approx(1.0)
    assert moe.last_stats.capacity_per_expert > 0
    assert moe.last_stats.dropped_routes >= 0


def test_recurrent_stats_compatibility() -> None:
    config = make_config(n_layers=1, recurrent_depth=3)
    recurrent = RecurrentRefinement(config)
    output = recurrent(torch.randn(2, 4, config.d_model))
    assert output.shape == (2, 4, config.d_model)
    assert recurrent.last_stats is not None
    assert recurrent.last_stats.depth == 3
