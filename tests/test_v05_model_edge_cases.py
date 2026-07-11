from __future__ import annotations

import pytest
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.model import ModelOutput, ReasonerCore


def _core(**changes) -> ReasonerCore:
    values = {
        "vocab_size": 32,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_ff": 32,
        "max_seq_len": 16,
        "seed": 6,
    }
    values.update(changes)
    return ReasonerCore(ReasonerConfig.from_dict(values))


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads", "mla_latent_dim"),
    [("mha", 4, 12), ("gqa", 2, 12), ("mla_lite", 2, 4)],
)
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padded_cached_and_uncached_generation_match(
    attention_type: str,
    n_kv_heads: int,
    mla_latent_dim: int,
    padding_side: str,
) -> None:
    model = _core(
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        mla_latent_dim=mla_latent_dim,
    ).transformer
    if padding_side == "left":
        prompt = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool)
    else:
        prompt = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    cached = model.generate(prompt, 2, attention_mask=mask, use_cache=True)
    uncached = model.generate(prompt, 2, attention_mask=mask, use_cache=False)
    assert torch.equal(cached, uncached)


def test_generation_sampling_eos_and_validation() -> None:
    model = _core().transformer
    prompt = torch.tensor([[1, 2]])
    assert torch.equal(model.generate(prompt, 0), prompt)
    generator = torch.Generator().manual_seed(3)
    sampled = model.generate(
        prompt,
        2,
        do_sample=True,
        temperature=0.7,
        top_k=3,
        generator=generator,
    )
    assert sampled.shape == (1, 4)
    with pytest.raises(ValueError, match="shape"):
        model.generate(torch.tensor([1, 2]), 1)
    with pytest.raises(ValueError, match="negative"):
        model.generate(prompt, -1)
    with pytest.raises(ValueError, match="exceeds"):
        model.generate(torch.ones((1, 16), dtype=torch.long), 1)
    with pytest.raises(ValueError, match="vocabulary"):
        model.generate(prompt, 1, eos_token_id=99)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, 1, do_sample=True, temperature=0)
    with pytest.raises(ValueError, match="top_k"):
        model.generate(prompt, 1, do_sample=True, top_k=0)
    with pytest.raises(ValueError, match="at least one"):
        model.generate(prompt, 1, attention_mask=torch.zeros_like(prompt, dtype=torch.bool))


def test_forward_state_hidden_and_memory_validation() -> None:
    core = _core(recurrent_depth=2)
    tokens = torch.tensor([[1, 2, 3]])
    output = core.transformer(tokens, return_hidden=True, return_dict=True)
    assert isinstance(output, ModelOutput)
    assert output.hidden_states is not None
    with pytest.raises(ValueError, match="shape"):
        core.transformer(torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="empty"):
        core.transformer(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="batch_size"):
        core(tokens, state=core.init_state(2))
    with pytest.raises(ValueError, match="cached decoding"):
        core(tokens, mode="think", use_cache=True)
    with pytest.raises(ValueError, match="positive"):
        core.think(tokens, 0)
    with pytest.raises(ValueError, match="positive"):
        core.estimate_memory_cost(0, 1)
    inspected = core.inspect_activations(tokens)
    assert inspected["finite"] is True
    assert inspected["recurrent"] is not None


def test_layer_cache_rejects_inconsistent_metadata_batches_and_dtypes() -> None:
    from wai_r0.model import LayerKVCache

    key = torch.randn(2, 1, 3, 4)
    value = torch.randn(2, 1, 3, 4)
    with pytest.raises(ValueError, match="position_ids batch"):
        LayerKVCache(key=key, value=value, position_ids=torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="integer dtype"):
        LayerKVCache(key=key, value=value, position_ids=torch.zeros(2, 3))
    with pytest.raises(ValueError, match="mask batch"):
        LayerKVCache(key=key, value=value, key_padding_mask=torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool dtype"):
        LayerKVCache(key=key, value=value, key_padding_mask=torch.ones(2, 3, dtype=torch.long))


def test_core_state_preserves_full_cached_mask_with_query_only_updates() -> None:
    config = ReasonerConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_seq_len=12,
        attention_type="gqa",
    )
    core = ReasonerCore(config)
    state = core.init_state(1)
    prompt = torch.tensor([[1, 2, 0]])
    prompt_mask = torch.tensor([[True, True, False]])
    core(prompt, state=state, attention_mask=prompt_mask, use_cache=True, return_dict=True)
    core(
        torch.tensor([[3]]),
        state=state,
        attention_mask=torch.tensor([[True]]),
        use_cache=True,
        return_dict=True,
    )
    assert state["attention_mask"].tolist() == [[True, True, False, True]]


def test_generation_stops_when_every_batch_item_emits_eos() -> None:
    model = _core().transformer
    with torch.no_grad():
        model.head.weight.zero_()
    prompt = torch.tensor([[1, 2], [3, 4]])
    generated = model.generate(prompt, 5, eos_token_id=0)
    assert generated.shape == (2, 3)
    assert generated[:, -1].eq(0).all()
