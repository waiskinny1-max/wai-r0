from __future__ import annotations

import random

import pytest
import torch

from wai_r0.modeling.common import (
    RMSNorm,
    RotaryEmbedding,
    apply_rope,
    dtype_from_name,
    repeat_kv,
    rope_cache,
    set_seed,
    temporary_seed,
)


def test_seed_and_temporary_seed_restore_rng() -> None:
    set_seed(9)
    expected_before = random.random()
    expected_torch_before = torch.rand(2)
    with temporary_seed(99):
        inside = (random.random(), torch.rand(2))
    after = (random.random(), torch.rand(2))

    set_seed(9)
    assert random.random() == expected_before
    torch.testing.assert_close(torch.rand(2), expected_torch_before, rtol=0, atol=0)
    expected_after = (random.random(), torch.rand(2))
    assert after[0] == expected_after[0]
    torch.testing.assert_close(after[1], expected_after[1], rtol=0, atol=0)
    set_seed(99)
    assert inside[0] == random.random()


def test_dtype_aliases_and_rms_norm() -> None:
    assert dtype_from_name("fp32") == torch.float32
    assert dtype_from_name("FP16") == torch.float16
    assert dtype_from_name("bf16") == torch.bfloat16
    with pytest.raises(ValueError, match="unsupported"):
        dtype_from_name("int8")
    with pytest.raises(ValueError):
        RMSNorm(0)
    output = RMSNorm(4)(torch.randn(2, 3, 4))
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()


def test_rotary_embedding_validates_positions_and_caches() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(3, 8)
    with pytest.raises(ValueError, match="invalid"):
        RotaryEmbedding(4, 0)
    rope = RotaryEmbedding(4, 8)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    cos, sin = rope(positions, dtype=torch.float32)
    assert cos.shape == sin.shape == (1, 1, 3, 2)
    with pytest.raises(ValueError, match="shape"):
        rope(torch.tensor([0, 1]), dtype=torch.float32)
    with pytest.raises(ValueError, match="empty"):
        rope(torch.empty((1, 0), dtype=torch.long), dtype=torch.float32)
    with pytest.raises(ValueError, match="negative"):
        rope(torch.tensor([[-1]]), dtype=torch.float32)
    with pytest.raises(ValueError, match="max_seq_len"):
        rope(torch.tensor([[8]]), dtype=torch.float32)


def test_rope_helpers_and_repeat_kv_validation() -> None:
    device = torch.device("cpu")
    cos, sin = rope_cache(3, 4, device, torch.float32, offset=2)
    assert cos.shape == sin.shape == (3, 2)
    x = torch.randn(1, 2, 3, 4)
    rotated = apply_rope(x, cos, sin)
    assert rotated.shape == x.shape
    batched_cos = cos.unsqueeze(0)
    assert apply_rope(x, batched_cos, sin.unsqueeze(0)).shape == x.shape
    assert (
        apply_rope(x, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)).shape == x.shape
    )
    with pytest.raises(ValueError, match="even"):
        rope_cache(2, 3, device, torch.float32)
    with pytest.raises(ValueError, match="positive"):
        rope_cache(0, 4, device, torch.float32)
    with pytest.raises(ValueError, match="offset"):
        rope_cache(2, 4, device, torch.float32, offset=-1)
    with pytest.raises(ValueError, match="identical"):
        apply_rope(x, cos, sin[:2])
    with pytest.raises(ValueError, match="cache must"):
        apply_rope(x, torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="batch"):
        apply_rope(x, cos.expand(2, -1, -1), sin.expand(2, -1, -1))
    assert repeat_kv(x[:, :1], 2).shape[1] == 2
    assert repeat_kv(x, 1) is x
    with pytest.raises(ValueError):
        repeat_kv(x, 0)
    with pytest.raises(ValueError, match="shape"):
        repeat_kv(torch.randn(2, 3), 2)
