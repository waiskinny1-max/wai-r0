from __future__ import annotations

import torch
from torch import nn

from wai_r0.config import ReasonerConfig
from wai_r0.modeling.common import RMSNorm
from wai_r0.modeling.feedforward import SwiGLU
from wai_r0.modeling.types import RecurrentStats


class RecurrentRefinement(nn.Module):
    """Weight-tied latent refinement with fixed, drift, or learned halting."""

    def __init__(self, cfg: ReasonerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.d_model, cfg.norm_epsilon)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.halt_projection = (
            nn.Linear(cfg.d_model, 1, bias=True) if cfg.recurrent_halt_mode == "learned" else None
        )
        self.last_stats: RecurrentStats | None = None
        self.last_auxiliary_losses: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        steps: int | None = None,
        collect_diagnostics: bool | None = None,
    ) -> torch.Tensor:
        requested_steps = self.cfg.recurrent_depth if steps is None else steps
        if requested_steps < 1:
            raise ValueError("recurrent steps must be positive")
        minimum_steps = min(self.cfg.recurrent_min_steps, requested_steps)
        collect = (
            self.cfg.diagnostics_default if collect_diagnostics is None else collect_diagnostics
        )

        state = x
        norms: list[torch.Tensor] = []
        drifts: list[torch.Tensor] = []
        halt_probabilities: list[torch.Tensor] = []
        halted_early = False
        ponder_terms: list[torch.Tensor] = []

        for step_index in range(requested_steps):
            previous = state
            state = state + self.ff(self.norm(state))
            drift = (state - previous).float().norm(dim=-1).mean()
            state_norm = state.float().norm(dim=-1).mean()
            if collect or self.cfg.recurrent_halt_mode != "fixed":
                norms.append(state_norm)
                drifts.append(drift)

            should_stop = False
            if self.cfg.recurrent_halt_mode == "drift":
                threshold = self.cfg.recurrent_halt_threshold
                if threshold is None:
                    raise RuntimeError("drift halting threshold is missing")
                should_stop = step_index + 1 >= minimum_steps and bool(
                    (drift <= threshold).detach().cpu()
                )
            elif self.cfg.recurrent_halt_mode == "learned":
                if self.halt_projection is None or self.cfg.recurrent_halt_threshold is None:
                    raise RuntimeError("learned halting is not configured")
                probability = torch.sigmoid(self.halt_projection(self.norm(state))).mean()
                halt_probabilities.append(probability)
                ponder_terms.append(1.0 - probability)
                should_stop = step_index + 1 >= minimum_steps and bool(
                    (probability >= self.cfg.recurrent_halt_threshold).detach().cpu()
                )

            if should_stop:
                halted_early = True
                break

        ponder_loss: torch.Tensor | None = None
        if ponder_terms:
            ponder_loss = torch.stack(ponder_terms).mean() * self.cfg.recurrent_ponder_loss_coef
            self.last_auxiliary_losses = {"recurrent_ponder": ponder_loss}
        else:
            self.last_auxiliary_losses = {}

        self.last_stats = RecurrentStats(
            depth=step_index + 1,
            _norm_by_step=norms,
            _drift_by_step=drifts,
            halted_early=halted_early,
            halt_mode=self.cfg.recurrent_halt_mode,
            _halt_probability_by_step=halt_probabilities,
            _ponder_loss=ponder_loss,
        )
        return state
