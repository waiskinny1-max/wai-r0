from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from wai_r0.config import ReasonerConfig
from wai_r0.modeling.common import RotaryEmbedding, apply_rope, repeat_kv
from wai_r0.modeling.types import AttentionStats, LayerKVCache


def _default_position_ids(
    *,
    batch: int,
    query_len: int,
    past_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 2:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.shape[0] != batch:
            raise ValueError("attention_mask batch dimension does not match input")
        if mask.shape[1] == past_len + query_len:
            positions = mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
            return positions[:, -query_len:]
        if past_len == 0 and mask.shape[1] == query_len:
            return mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
    return torch.arange(past_len, past_len + query_len, device=device).expand(batch, -1)


def _normalize_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch: int,
    query_len: int,
    past_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if position_ids is None:
        return _default_position_ids(
            batch=batch,
            query_len=query_len,
            past_len=past_len,
            device=device,
            attention_mask=attention_mask,
        )
    if position_ids.shape != (batch, query_len):
        raise ValueError("position_ids must have shape [batch, query_len]")
    return position_ids.to(device=device, dtype=torch.long)


def _full_key_padding_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    query_len: int,
    past_len: int,
    device: torch.device,
    cached_mask: torch.Tensor | None,
) -> torch.Tensor:
    key_len = past_len + query_len
    if attention_mask is None or attention_mask.ndim != 2:
        current = torch.ones(batch, query_len, device=device, dtype=torch.bool)
        if past_len == 0:
            return current
        previous = (
            cached_mask.to(device=device, dtype=torch.bool)
            if cached_mask is not None
            else torch.ones(batch, past_len, device=device, dtype=torch.bool)
        )
        return torch.cat((previous, current), dim=1)

    mask = attention_mask.to(device=device, dtype=torch.bool)
    if mask.shape == (batch, key_len):
        return mask
    if mask.shape == (batch, query_len):
        if past_len == 0:
            return mask
        previous = (
            cached_mask.to(device=device, dtype=torch.bool)
            if cached_mask is not None
            else torch.ones(batch, past_len, device=device, dtype=torch.bool)
        )
        return torch.cat((previous, mask), dim=1)
    raise ValueError("2D attention_mask must have shape [batch, query_len] or [batch, key_len]")


def _attention_permissions(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    query_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
    key_padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_positions = torch.arange(query_len, device=device) + past_len
    key_positions = torch.arange(key_len, device=device)
    causal = key_positions[None, :] <= query_positions[:, None]
    allowed = causal[None].expand(batch, -1, -1) & key_padding_mask[:, None, :]

    if attention_mask is not None and attention_mask.ndim in (3, 4):
        relation = attention_mask
        if relation.ndim == 4:
            if relation.shape[1] != 1:
                raise ValueError("4D attention_mask must have a singleton head dimension")
            relation = relation[:, 0]
        if relation.shape != (batch, query_len, key_len):
            raise ValueError("relation attention_mask must have shape [B,Q,K]")
        allowed &= relation.to(device=device, dtype=torch.bool)

    query_mask = key_padding_mask[:, -query_len:]
    allowed &= query_mask[:, :, None]
    return allowed, query_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ReasonerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hd = cfg.head_dim
        self.last_stats: AttentionStats | None = None
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=cfg.qkv_bias)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=cfg.qkv_bias)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=cfg.qkv_bias)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.hd, cfg.max_seq_len, cfg.rope_base)

    def estimate_kv_cache_bytes(self, seq_len: int, batch: int, dtype: torch.dtype) -> int:
        if seq_len < 0 or batch < 1:
            raise ValueError("invalid cache dimensions")
        element_size = int(torch.empty((), dtype=dtype).element_size())
        return 2 * batch * seq_len * self.cfg.n_kv_heads * self.hd * element_size

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        allowed: torch.Tensor,
        query_mask: torch.Tensor,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.cfg.use_sdpa and not collect_diagnostics:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed[:, None],
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=False,
            )
            output = output * query_mask[:, None, :, None].to(dtype=output.dtype)
            return output, None

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.hd**-0.5)
        expanded_allowed = allowed[:, None]
        has_keys = expanded_allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~expanded_allowed, float("-inf"))
        scores = torch.where(has_keys, scores, torch.zeros_like(scores))
        probabilities = scores.softmax(dim=-1)
        probabilities = torch.where(has_keys, probabilities, torch.zeros_like(probabilities))
        probabilities = self.dropout(probabilities.to(dtype=q.dtype))
        output = torch.matmul(probabilities, v)
        output = output * query_mask[:, None, :, None].to(dtype=output.dtype)
        entropy = (
            -(probabilities.float() * probabilities.float().clamp_min(1e-12).log())
            .sum(dim=-1)
            .masked_select(query_mask[:, None, :])
        )
        mean_entropy = entropy.mean() if entropy.numel() else torch.zeros((), device=q.device)
        return output, mean_entropy

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
        collect_diagnostics: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerKVCache | None]:
        batch, time, _ = x.shape
        past_len = past_key_value.sequence_length if past_key_value is not None else 0
        if past_len + time > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if past_key_value is not None and past_key_value.batch_size != batch:
            raise ValueError("cache batch size does not match input")
        collect = (
            self.cfg.diagnostics_default if collect_diagnostics is None else collect_diagnostics
        )
        positions = _normalize_position_ids(
            position_ids,
            batch=batch,
            query_len=time,
            past_len=past_len,
            device=x.device,
            attention_mask=attention_mask,
        )

        q = self.q(x).view(batch, time, self.cfg.n_heads, self.hd).transpose(1, 2)
        k_new = self.k(x).view(batch, time, self.cfg.n_kv_heads, self.hd).transpose(1, 2)
        v_new = self.v(x).view(batch, time, self.cfg.n_kv_heads, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k_new = apply_rope(k_new, cos, sin)

        if past_key_value is not None:
            if past_key_value.key is None or past_key_value.value is None:
                raise ValueError("dense attention received an incompatible cache")
            k_all = torch.cat((past_key_value.key, k_new), dim=2)
            v_all = torch.cat((past_key_value.value, v_new), dim=2)
        else:
            k_all = k_new
            v_all = v_new

        key_padding_mask = _full_key_padding_mask(
            attention_mask,
            batch=batch,
            query_len=time,
            past_len=past_len,
            device=x.device,
            cached_mask=past_key_value.key_padding_mask if past_key_value is not None else None,
        )
        allowed, query_mask = _attention_permissions(
            attention_mask,
            batch=batch,
            query_len=time,
            key_len=past_len + time,
            past_len=past_len,
            device=x.device,
            key_padding_mask=key_padding_mask,
        )
        expanded_k = repeat_kv(k_all, self.cfg.n_heads // self.cfg.n_kv_heads)
        expanded_v = repeat_kv(v_all, self.cfg.n_heads // self.cfg.n_kv_heads)
        attended, entropy = self._compute_attention(
            q,
            expanded_k,
            expanded_v,
            allowed=allowed,
            query_mask=query_mask,
            collect_diagnostics=collect,
        )
        output = attended.transpose(1, 2).contiguous().view(batch, time, self.cfg.d_model)
        present = (
            LayerKVCache(key=k_all, value=v_all, key_padding_mask=key_padding_mask)
            if use_cache
            else None
        )
        payload_bytes = (
            present.payload_bytes
            if present is not None
            else self.estimate_kv_cache_bytes(past_len + time, batch, x.dtype)
        )
        total_bytes = present.allocated_bytes if present is not None else payload_bytes
        masked_fraction = 0.0
        if collect:
            masked_fraction = float((~query_mask).float().mean().detach().cpu())
        self.last_stats = AttentionStats(
            _attention_entropy=entropy,
            kv_cache_bytes=total_bytes,
            payload_cache_bytes=payload_bytes,
            query_tokens=time,
            key_tokens=past_len + time,
            masked_query_fraction=masked_fraction,
        )
        result = self.o(output)
        return (result, present) if use_cache else result


class MLALiteAttention(CausalSelfAttention):
    """MLA-inspired compressed latent cache, not a DeepSeek MLA reproduction.

    K/V are reconstructed from the latent cache at each decode step. This lowers
    cache memory but can increase decode compute; the profiler reports both.
    """

    def __init__(self, cfg: ReasonerConfig) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        self.hd = cfg.head_dim
        self.last_stats: AttentionStats | None = None
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=cfg.qkv_bias)
        self.down = nn.Linear(cfg.d_model, cfg.mla_latent_dim, bias=False)
        self.kup = nn.Linear(cfg.mla_latent_dim, cfg.n_kv_heads * self.hd, bias=cfg.qkv_bias)
        self.vup = nn.Linear(cfg.mla_latent_dim, cfg.n_kv_heads * self.hd, bias=cfg.qkv_bias)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.hd, cfg.max_seq_len, cfg.rope_base)

    def estimate_kv_cache_bytes(self, seq_len: int, batch: int, dtype: torch.dtype) -> int:
        if seq_len < 0 or batch < 1:
            raise ValueError("invalid cache dimensions")
        element_size = int(torch.empty((), dtype=dtype).element_size())
        return batch * seq_len * self.cfg.mla_latent_dim * element_size

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
        collect_diagnostics: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerKVCache | None]:
        batch, time, _ = x.shape
        past_len = past_key_value.sequence_length if past_key_value is not None else 0
        if past_len + time > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if past_key_value is not None and past_key_value.batch_size != batch:
            raise ValueError("cache batch size does not match input")
        collect = (
            self.cfg.diagnostics_default if collect_diagnostics is None else collect_diagnostics
        )
        positions_new = _normalize_position_ids(
            position_ids,
            batch=batch,
            query_len=time,
            past_len=past_len,
            device=x.device,
            attention_mask=attention_mask,
        )

        q = self.q(x).view(batch, time, self.cfg.n_heads, self.hd).transpose(1, 2)
        latent_new = self.down(x)
        if past_key_value is not None:
            if past_key_value.latent is None:
                raise ValueError("MLA-lite attention received an incompatible cache")
            latent_all = torch.cat((past_key_value.latent, latent_new), dim=1)
            previous_positions = past_key_value.position_ids
            if previous_positions is None:
                previous_positions = torch.arange(past_len, device=x.device).expand(batch, -1)
            positions_all = torch.cat((previous_positions, positions_new), dim=1)
        else:
            latent_all = latent_new
            positions_all = positions_new

        total_len = latent_all.shape[1]
        k_all = (
            self.kup(latent_all)
            .view(batch, total_len, self.cfg.n_kv_heads, self.hd)
            .transpose(1, 2)
        )
        v_all = (
            self.vup(latent_all)
            .view(batch, total_len, self.cfg.n_kv_heads, self.hd)
            .transpose(1, 2)
        )
        q_cos, q_sin = self.rope(positions_new, dtype=x.dtype)
        k_cos, k_sin = self.rope(positions_all, dtype=x.dtype)
        q = apply_rope(q, q_cos, q_sin)
        k_all = apply_rope(k_all, k_cos, k_sin)

        key_padding_mask = _full_key_padding_mask(
            attention_mask,
            batch=batch,
            query_len=time,
            past_len=past_len,
            device=x.device,
            cached_mask=past_key_value.key_padding_mask if past_key_value is not None else None,
        )
        allowed, query_mask = _attention_permissions(
            attention_mask,
            batch=batch,
            query_len=time,
            key_len=total_len,
            past_len=past_len,
            device=x.device,
            key_padding_mask=key_padding_mask,
        )
        expanded_k = repeat_kv(k_all, self.cfg.n_heads // self.cfg.n_kv_heads)
        expanded_v = repeat_kv(v_all, self.cfg.n_heads // self.cfg.n_kv_heads)
        attended, entropy = self._compute_attention(
            q,
            expanded_k,
            expanded_v,
            allowed=allowed,
            query_mask=query_mask,
            collect_diagnostics=collect,
        )
        output = attended.transpose(1, 2).contiguous().view(batch, time, self.cfg.d_model)
        present = (
            LayerKVCache(
                latent=latent_all,
                position_ids=positions_all,
                key_padding_mask=key_padding_mask,
            )
            if use_cache
            else None
        )
        latent_payload_bytes = (
            present.payload_bytes
            if present is not None
            else self.estimate_kv_cache_bytes(total_len, batch, x.dtype)
        )
        total_bytes = present.allocated_bytes if present is not None else latent_payload_bytes
        dense_bytes = (
            2
            * batch
            * total_len
            * self.cfg.n_kv_heads
            * self.hd
            * torch.empty((), dtype=x.dtype).element_size()
        )
        masked_fraction = 0.0
        if collect:
            masked_fraction = float((~query_mask).float().mean().detach().cpu())
        self.last_stats = AttentionStats(
            _attention_entropy=entropy,
            kv_cache_bytes=total_bytes,
            payload_cache_bytes=latent_payload_bytes,
            compression_ratio=latent_payload_bytes / max(1, dense_bytes),
            query_tokens=time,
            key_tokens=total_len,
            masked_query_fraction=masked_fraction,
        )
        result = self.o(output)
        return (result, present) if use_cache else result
