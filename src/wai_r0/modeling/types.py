from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from wai_r0.modeling.common import scalar_to_float, tensor_to_float_list


@dataclass(slots=True)
class LayerKVCache:
    """One layer's autoregressive cache.

    Dense attention stores rotated K/V tensors. MLA-lite stores only the latent
    representation and its position IDs, preserving the intended memory tradeoff.
    """

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        dense = self.key is not None or self.value is not None
        compressed = self.latent is not None
        if dense and compressed:
            raise ValueError("cache cannot contain both dense K/V and MLA latent state")
        if dense and (self.key is None or self.value is None):
            raise ValueError("dense cache requires both key and value")
        if not dense and not compressed:
            raise ValueError("cache must contain dense K/V or MLA latent state")
        lengths = []
        if self.key is not None:
            if self.key.ndim != 4 or self.value is None or self.value.shape != self.key.shape:
                raise ValueError("dense cache tensors must have matching [B,H,T,D] shapes")
            lengths.append(self.key.shape[2])
        if self.latent is not None:
            if self.latent.ndim != 3:
                raise ValueError("latent cache must have shape [B,T,D]")
            lengths.append(self.latent.shape[1])
        if self.position_ids is not None:
            if self.position_ids.ndim != 2:
                raise ValueError("position_ids must have shape [B,T]")
            lengths.append(self.position_ids.shape[1])
        payload = self.key if self.key is not None else self.latent
        if payload is None:
            raise RuntimeError("cache payload is unexpectedly absent")
        batch_size = payload.shape[0]
        if self.position_ids is not None:
            if self.position_ids.shape[0] != batch_size:
                raise ValueError("cache position_ids batch dimension is inconsistent")
            if self.position_ids.dtype not in {torch.int32, torch.int64}:
                raise ValueError("cache position_ids must use an integer dtype")
        if self.key_padding_mask is not None:
            if self.key_padding_mask.ndim != 2:
                raise ValueError("key_padding_mask must have shape [B,T]")
            if self.key_padding_mask.shape[0] != batch_size:
                raise ValueError("cache key_padding_mask batch dimension is inconsistent")
            if self.key_padding_mask.dtype != torch.bool:
                raise ValueError("cache key_padding_mask must use bool dtype")
            lengths.append(self.key_padding_mask.shape[1])
        if len(set(lengths)) > 1:
            raise ValueError("cache components have inconsistent sequence lengths")

    @property
    def sequence_length(self) -> int:
        if self.key is not None:
            return int(self.key.shape[2])
        if self.latent is not None:
            return int(self.latent.shape[1])
        raise RuntimeError("invalid empty cache")

    @property
    def batch_size(self) -> int:
        tensor = self.key if self.key is not None else self.latent
        if tensor is None:
            raise RuntimeError("invalid empty cache")
        return int(tensor.shape[0])

    @property
    def allocated_bytes(self) -> int:
        tensors = (self.key, self.value, self.latent, self.position_ids, self.key_padding_mask)
        return int(sum(item.numel() * item.element_size() for item in tensors if item is not None))

    @property
    def payload_bytes(self) -> int:
        """Bytes for model state only, excluding masks and integer positions."""

        tensors = (self.key, self.value, self.latent)
        return int(sum(item.numel() * item.element_size() for item in tensors if item is not None))

    @property
    def metadata_bytes(self) -> int:
        """Bytes used by positions and masks rather than model payload."""

        tensors = (self.position_ids, self.key_padding_mask)
        return int(sum(item.numel() * item.element_size() for item in tensors if item is not None))

    @property
    def cache_kind(self) -> str:
        return "mla_latent" if self.latent is not None else "dense_kv"

    def detached(self) -> LayerKVCache:
        def detach(value: torch.Tensor | None) -> torch.Tensor | None:
            return value.detach() if value is not None else None

        return LayerKVCache(
            key=detach(self.key),
            value=detach(self.value),
            latent=detach(self.latent),
            position_ids=detach(self.position_ids),
            key_padding_mask=detach(self.key_padding_mask),
        )


@dataclass(slots=True)
class AttentionStats:
    _attention_entropy: torch.Tensor | None
    kv_cache_bytes: int
    payload_cache_bytes: int | None = None
    compression_ratio: float | None = None
    query_tokens: int = 0
    key_tokens: int = 0
    masked_query_fraction: float = 0.0

    @property
    def attention_entropy(self) -> float:
        return 0.0 if self._attention_entropy is None else scalar_to_float(self._attention_entropy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_entropy": self.attention_entropy,
            "kv_cache_bytes": int(self.kv_cache_bytes),
            "payload_cache_bytes": int(self.payload_cache_bytes or self.kv_cache_bytes),
            "compression_ratio": self.compression_ratio,
            "query_tokens": int(self.query_tokens),
            "key_tokens": int(self.key_tokens),
            "masked_query_fraction": float(self.masked_query_fraction),
        }


@dataclass(slots=True)
class MoEStats:
    _router_entropy: torch.Tensor
    _load_fraction: torch.Tensor
    _accepted_load_fraction: torch.Tensor
    _dropped_routes: torch.Tensor
    capacity_per_expert: int
    _route_count: torch.Tensor
    _accepted_route_count: torch.Tensor

    @property
    def router_entropy(self) -> float:
        return scalar_to_float(self._router_entropy)

    @property
    def load_fraction(self) -> list[float]:
        return tensor_to_float_list(self._load_fraction)

    @property
    def accepted_load_fraction(self) -> list[float]:
        return tensor_to_float_list(self._accepted_load_fraction)

    @property
    def dropped_routes(self) -> int:
        return int(self._dropped_routes.detach().cpu())

    @property
    def route_count(self) -> int:
        return int(self._route_count.detach().cpu())

    @property
    def accepted_route_count(self) -> int:
        return int(self._accepted_route_count.detach().cpu())

    @property
    def dropped_route_fraction(self) -> float:
        return self.dropped_routes / max(1, self.route_count)

    @property
    def collapse_warning(self) -> bool:
        fractions = self.load_fraction
        return bool(fractions and max(fractions) > 0.90)

    def to_dict(self) -> dict[str, Any]:
        return {
            "router_entropy": self.router_entropy,
            "load_fraction": self.load_fraction,
            "accepted_load_fraction": self.accepted_load_fraction,
            "collapse_warning": self.collapse_warning,
            "dropped_routes": int(self.dropped_routes),
            "dropped_route_fraction": self.dropped_route_fraction,
            "capacity_per_expert": int(self.capacity_per_expert),
            "route_count": int(self.route_count),
            "accepted_route_count": int(self.accepted_route_count),
        }


@dataclass(slots=True)
class RecurrentStats:
    depth: int
    _norm_by_step: list[torch.Tensor]
    _drift_by_step: list[torch.Tensor]
    halted_early: bool = False
    halt_mode: str = "fixed"
    _halt_probability_by_step: list[torch.Tensor] = field(default_factory=list)
    _ponder_loss: torch.Tensor | None = None

    @property
    def norm_by_step(self) -> list[float]:
        return [scalar_to_float(item) for item in self._norm_by_step]

    @property
    def drift_by_step(self) -> list[float]:
        return [scalar_to_float(item) for item in self._drift_by_step]

    @property
    def halt_probability_by_step(self) -> list[float]:
        return [scalar_to_float(item) for item in self._halt_probability_by_step]

    @property
    def ponder_loss(self) -> float | None:
        return None if self._ponder_loss is None else scalar_to_float(self._ponder_loss)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": int(self.depth),
            "norm_by_step": self.norm_by_step,
            "drift_by_step": self.drift_by_step,
            "halted_early": bool(self.halted_early),
            "halt_mode": self.halt_mode,
            "halt_probability_by_step": self.halt_probability_by_step,
            "ponder_loss": self.ponder_loss,
        }


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    past_key_values: tuple[LayerKVCache, ...] | None = None
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_bytes(self) -> int:
        return int(sum(cache.allocated_bytes for cache in self.past_key_values or ()))

    @property
    def cache_payload_bytes(self) -> int:
        return int(sum(cache.payload_bytes for cache in self.past_key_values or ()))
