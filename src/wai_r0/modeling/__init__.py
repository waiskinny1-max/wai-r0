"""Modular model implementation used by the compatibility facade."""

from wai_r0.modeling.attention import CausalSelfAttention, MLALiteAttention
from wai_r0.modeling.common import (
    RMSNorm,
    apply_rope,
    dtype_from_name,
    repeat_kv,
    rope_cache,
    set_seed,
)
from wai_r0.modeling.feedforward import SwiGLU, TopKMoE
from wai_r0.modeling.recurrence import RecurrentRefinement
from wai_r0.modeling.transformer import Block, DecoderOnlyTransformer, ReasonerCore
from wai_r0.modeling.types import (
    AttentionStats,
    LayerKVCache,
    ModelOutput,
    MoEStats,
    RecurrentStats,
)

__all__ = [
    "AttentionStats",
    "Block",
    "CausalSelfAttention",
    "DecoderOnlyTransformer",
    "LayerKVCache",
    "MLALiteAttention",
    "MoEStats",
    "ModelOutput",
    "RMSNorm",
    "ReasonerCore",
    "RecurrentRefinement",
    "RecurrentStats",
    "SwiGLU",
    "TopKMoE",
    "apply_rope",
    "dtype_from_name",
    "repeat_kv",
    "rope_cache",
    "set_seed",
]
