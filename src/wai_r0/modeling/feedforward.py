from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from wai_r0.config import ReasonerConfig
from wai_r0.modeling.types import MoEStats


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model < 1 or d_ff < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))


class TopKMoE(nn.Module):
    """Small capacity-limited top-k MoE for controlled local experiments.

    This implementation prioritizes auditability over distributed throughput. It
    records raw and accepted load, dropped routes, router entropy, balancing loss,
    and z-loss. Dense controls remain necessary when interpreting results.
    """

    def __init__(self, cfg: ReasonerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_experts)]
        )
        self.shared = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout) if cfg.moe_shared_expert else None
        self.last_stats: MoEStats | None = None
        self.last_auxiliary_losses: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim < 2 or x.shape[-1] != self.cfg.d_model:
            raise ValueError("MoE input must end in d_model")
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat).float()
        probabilities = router_logits.softmax(dim=-1)
        weights, indices = torch.topk(
            probabilities,
            self.cfg.experts_per_token,
            dim=-1,
        )
        weights = weights.to(dtype=flat.dtype)
        if self.cfg.moe_normalize_topk:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        route_count = flat.shape[0] * self.cfg.experts_per_token
        capacity = max(
            self.cfg.moe_min_capacity,
            math.ceil(self.cfg.moe_capacity_factor * route_count / self.cfg.n_experts),
        )
        accepted = torch.zeros_like(indices, dtype=torch.bool)
        raw_load = torch.bincount(
            indices.reshape(-1),
            minlength=self.cfg.n_experts,
        ).to(dtype=torch.float32)

        for expert_id in range(self.cfg.n_experts):
            token_index, route_index = indices.eq(expert_id).nonzero(as_tuple=True)
            if token_index.numel() == 0:
                continue
            route_weights = weights[token_index, route_index]
            if token_index.numel() > capacity:
                selected = torch.topk(route_weights.float(), capacity, sorted=False).indices
                token_index = token_index[selected]
                route_index = route_index[selected]
            accepted[token_index, route_index] = True

        dispatch_weights = weights * accepted.to(dtype=weights.dtype)
        if self.cfg.moe_normalize_topk:
            dispatch_weights = dispatch_weights / dispatch_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        output = torch.zeros_like(flat)
        accepted_load = torch.zeros(
            self.cfg.n_experts,
            device=flat.device,
            dtype=torch.float32,
        )
        for expert_id, expert in enumerate(self.experts):
            token_index, route_index = (accepted & indices.eq(expert_id)).nonzero(as_tuple=True)
            if token_index.numel() == 0:
                continue
            accepted_load[expert_id] = token_index.numel()
            contribution = expert(flat[token_index])
            contribution = contribution * dispatch_weights[token_index, route_index].unsqueeze(-1)
            output.index_add_(0, token_index, contribution)

        if self.shared is not None:
            output = output + self.shared(flat)

        accepted_route_count = accepted.sum()
        route_count_tensor = torch.tensor(route_count, device=flat.device, dtype=torch.long)
        dropped_routes = route_count_tensor - accepted_route_count
        mean_probability = probabilities.mean(dim=0)
        dispatch_fraction = raw_load.to(mean_probability.device) / max(1, route_count)
        load_balance = self.cfg.n_experts * torch.sum(mean_probability * dispatch_fraction)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        auxiliary_losses = {
            "moe_load_balance": load_balance * self.cfg.moe_load_balance_coef,
            "moe_router_z": router_z_loss * self.cfg.moe_router_z_loss_coef,
        }
        self.last_auxiliary_losses = auxiliary_losses
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
        self.last_stats = MoEStats(
            _router_entropy=entropy,
            _load_fraction=raw_load / raw_load.sum().clamp_min(1),
            _accepted_load_fraction=accepted_load / accepted_load.sum().clamp_min(1),
            _dropped_routes=dropped_routes,
            capacity_per_expert=capacity,
            _route_count=route_count_tensor,
            _accepted_route_count=accepted_route_count,
        )
        result = output.reshape(shape)
        return (result, auxiliary_losses) if return_aux else result
