"""Compatibility facade for the modular model implementation.

Existing imports from ``wai_r0.model`` remain valid. New code may import from
``wai_r0.modeling`` when a narrower dependency is useful.
"""

from wai_r0.modeling import (
    AttentionStats,
    Block,
    CausalSelfAttention,
    DecoderOnlyTransformer,
    LayerKVCache,
    MLALiteAttention,
    ModelOutput,
    MoEStats,
    ReasonerCore,
    RecurrentRefinement,
    RecurrentStats,
    RMSNorm,
    SwiGLU,
    TopKMoE,
    apply_rope,
    dtype_from_name,
    repeat_kv,
    rope_cache,
    set_seed,
)
from wai_r0.modeling.common import (
    scalar_to_float as _scalar,
)
from wai_r0.modeling.common import (
    tensor_to_float_list as _float_list,
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
    "_float_list",
    "_scalar",
    "apply_rope",
    "dtype_from_name",
    "repeat_kv",
    "rope_cache",
    "set_seed",
]
