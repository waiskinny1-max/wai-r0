from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore, set_seed


@dataclass(frozen=True)
class PriorProbe:
    """One no-gradient architecture-prior diagnostic.

    These probes test mechanical properties of the architecture. They are not
    semantic tests and must not be reported as reasoning or language ability.
    """

    name: str
    score: float
    metrics: dict[str, Any]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded(value: float) -> float:
    if value != value:  # NaN check without importing math.
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def _random_tokens(cfg: ReasonerConfig, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    safe_len = min(seq_len, cfg.max_seq_len)
    if safe_len < 2:
        raise ValueError("architecture-prior diagnostics require seq_len >= 2")
    return torch.randint(1, cfg.vocab_size, (batch_size, safe_len), device=device)


@torch.no_grad()
def _hidden(core: ReasonerCore, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, hidden = core.transformer(tokens.to(core.device_obj), return_hidden=True)
    return logits, hidden


def activation_sanity_probe(cfg: ReasonerConfig, batch_size: int, seq_len: int) -> PriorProbe:
    set_seed(cfg.seed)
    core = ReasonerCore(cfg)
    tokens = _random_tokens(cfg, batch_size, seq_len, core.device_obj)
    logits, hidden = _hidden(core, tokens)
    norms = core.transformer.last_diagnostics.get("activation_norms", [])
    finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(hidden).all().item())
    min_norm = min(norms) if norms else 0.0
    max_norm = max(norms) if norms else 0.0
    norm_ratio = _safe_ratio(max_norm, min_norm) if min_norm > 0 else float("inf")
    score = 1.0 if finite and norm_ratio < 10.0 else 0.0
    return PriorProbe(
        name="activation_sanity",
        score=score,
        metrics={
            "finite_logits_and_hidden": finite,
            "activation_norms": [float(v) for v in norms],
            "activation_norm_ratio": float(norm_ratio),
            "hidden_norm_mean": float(hidden.float().norm(dim=-1).mean().item()),
        },
        limitations=["Numerical sanity only; no semantic content is being measured."],
    )


def positional_addressing_probe(cfg: ReasonerConfig, batch_size: int, seq_len: int) -> PriorProbe:
    set_seed(cfg.seed)
    core = ReasonerCore(cfg)
    safe_len = min(seq_len, cfg.max_seq_len)
    tokens = torch.ones((batch_size, safe_len), dtype=torch.long, device=core.device_obj)
    _, hidden = _hidden(core, tokens)
    normalized = F.normalize(hidden.float(), dim=-1)
    adjacent = (normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1)
    far = (normalized[:, :1] * normalized[:, -1:]).sum(dim=-1)
    spread = float(hidden.float().std(dim=1).mean().item())
    adjacent_cosine = float(adjacent.mean().item())
    far_cosine = float(far.mean().item())
    separation = abs(adjacent_cosine - far_cosine) + spread / max(1.0, float(cfg.d_model))
    score = _bounded(separation)
    return PriorProbe(
        name="positional_addressing",
        score=score,
        metrics={
            "repeated_token_id": 1,
            "adjacent_position_cosine": adjacent_cosine,
            "first_last_position_cosine": far_cosine,
            "position_hidden_spread": spread,
            "separation_proxy": float(separation),
        },
        limitations=[
            "This detects position-sensitive hidden-state variation, not position understanding.",
            "Random causal masking can create position effects even without useful algorithms.",
        ],
    )


def identity_signal_probe(cfg: ReasonerConfig, batch_size: int, seq_len: int) -> PriorProbe:
    set_seed(cfg.seed)
    core = ReasonerCore(cfg)
    tokens = _random_tokens(cfg, batch_size, seq_len, core.device_obj)
    _, hidden = _hidden(core, tokens)
    embedding = core.transformer.embed.weight.detach().float()
    scores = F.normalize(hidden.float(), dim=-1) @ F.normalize(embedding, dim=-1).T
    top1 = scores.argmax(dim=-1)
    target_rank_hits = (top1 == tokens).float().mean().item()
    target_scores = scores.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    non_target_mean = scores.masked_fill(F.one_hot(tokens, cfg.vocab_size).bool(), 0.0).sum(dim=-1) / max(1, cfg.vocab_size - 1)
    margin = float((target_scores - non_target_mean).mean().item())
    score = _bounded(float(target_rank_hits) + max(0.0, margin) * 0.25)
    return PriorProbe(
        name="identity_signal_preservation",
        score=score,
        metrics={
            "input_token_top1_retrieval": float(target_rank_hits),
            "target_vs_background_similarity_margin": margin,
        },
        limitations=[
            "This is an embedding/hidden-state signal probe, not a copy-task solution.",
            "Tied embeddings can inflate the proxy; compare against untied baselines if used seriously.",
        ],
    )


def memory_mechanics_probe(cfg: ReasonerConfig, batch_size: int, seq_len: int) -> PriorProbe:
    cfg_mha = ReasonerConfig.from_dict({**cfg.to_dict(), "attention_type": "mha", "n_kv_heads": cfg.n_heads})
    cfg_gqa = ReasonerConfig.from_dict({**cfg.to_dict(), "attention_type": "gqa", "n_kv_heads": max(1, cfg.n_heads // 2)})
    cfg_mla = ReasonerConfig.from_dict({**cfg.to_dict(), "attention_type": "mla_lite", "n_kv_heads": max(1, cfg.n_heads // 2)})
    mha = ReasonerCore(cfg_mha).estimate_memory_cost(seq_len, batch_size)["kv_cache_bytes"]
    gqa = ReasonerCore(cfg_gqa).estimate_memory_cost(seq_len, batch_size)["kv_cache_bytes"]
    mla = ReasonerCore(cfg_mla).estimate_memory_cost(seq_len, batch_size)["kv_cache_bytes"]
    best_ratio = min(_safe_ratio(gqa, mha), _safe_ratio(mla, mha))
    score = _bounded(1.0 - best_ratio)
    return PriorProbe(
        name="memory_mechanics",
        score=score,
        metrics={
            "mha_kv_cache_bytes": int(mha),
            "gqa_kv_cache_bytes": int(gqa),
            "mla_lite_latent_cache_bytes": int(mla),
            "gqa_over_mha": _safe_ratio(gqa, mha),
            "mla_lite_over_mha": _safe_ratio(mla, mha),
        },
        limitations=[
            "This is a static cache estimate, not a hardware profiler.",
            "MLA-lite cache reduction does not imply DeepSeek-style MLA equivalence or reasoning ability.",
        ],
    )


def recurrent_consistency_probe(cfg: ReasonerConfig, batch_size: int, seq_len: int, depths: tuple[int, ...]) -> PriorProbe:
    rows: list[dict[str, Any]] = []
    for depth in depths:
        if depth < 1:
            raise ValueError("recurrent depths must be positive")
        rcfg = ReasonerConfig.from_dict({**cfg.to_dict(), "recurrent_depth": depth})
        set_seed(rcfg.seed)
        core = ReasonerCore(rcfg)
        tokens = _random_tokens(rcfg, batch_size, seq_len, core.device_obj)
        _ = core(tokens, mode="think" if depth > 1 else "fast")
        stats = core.recurrent.last_stats.to_dict() if core.recurrent and core.recurrent.last_stats else None
        if stats:
            norms = [float(v) for v in stats["norm_by_step"]]
            drifts = [float(v) for v in stats["drift_by_step"]]
            rows.append(
                {
                    "depth": depth,
                    "evaluated": True,
                    "max_norm": max(norms) if norms else 0.0,
                    "final_norm": norms[-1] if norms else 0.0,
                    "mean_drift": sum(drifts) / max(1, len(drifts)),
                    "finite": all(torch.isfinite(torch.tensor(v)).item() for v in [*norms, *drifts]),
                }
            )
        else:
            rows.append({"depth": depth, "evaluated": False, "reason": "depth=1 has no recurrent refinement block"})
    evaluated = [row for row in rows if row.get("evaluated")]
    finite = all(bool(row.get("finite", False)) for row in evaluated) if evaluated else True
    max_norm = max((float(row.get("max_norm", 0.0)) for row in evaluated), default=0.0)
    score = 1.0 if finite and max_norm < 1e4 else 0.0
    if not evaluated:
        score = 0.5
    return PriorProbe(
        name="recurrent_consistency",
        score=score,
        metrics={"depths": rows, "max_observed_norm": max_norm, "all_evaluated_depths_finite": finite},
        limitations=[
            "The latent iterative refinement state is not a thought trace.",
            "Depth stability does not show task-solving ability without training/evaluation.",
        ],
    )


def routing_health_probe(cfg: ReasonerConfig, batch_size: int, seq_len: int) -> PriorProbe:
    if not cfg.use_moe:
        return PriorProbe(
            name="routing_health",
            score=0.5,
            metrics={"evaluated": False, "reason": "model config has use_moe=false"},
            limitations=["Run with a MoE config to evaluate router entropy and expert load."],
        )
    set_seed(cfg.seed)
    core = ReasonerCore(cfg)
    tokens = _random_tokens(cfg, batch_size, seq_len, core.device_obj)
    inspection = core.inspect_activations(tokens)
    moe_rows = inspection.get("diagnostics", {}).get("moe", [])
    entropies = [float(row.get("router_entropy", 0.0)) for row in moe_rows]
    max_loads = [max(row.get("load_fraction", [1.0])) for row in moe_rows]
    collapse = any(bool(row.get("collapse_warning", False)) for row in moe_rows)
    max_entropy = torch.log(torch.tensor(float(cfg.n_experts))).item() if cfg.n_experts > 1 else 1.0
    normalized_entropy = sum(entropies) / max(1, len(entropies)) / max(max_entropy, 1e-12)
    load_penalty = max(max_loads, default=1.0)
    score = _bounded(normalized_entropy * (1.0 - max(0.0, load_penalty - 0.5)))
    if collapse:
        score = min(score, 0.25)
    return PriorProbe(
        name="routing_health",
        score=score,
        metrics={
            "evaluated": True,
            "router_entropy": entropies,
            "normalized_router_entropy": float(normalized_entropy),
            "max_expert_load_fraction": float(max(max_loads, default=0.0)),
            "collapse_warning": collapse,
            "raw_moe_stats": moe_rows,
        },
        limitations=["Random router balance is not expert specialization; specialization requires training evidence."],
    )


def run_prior_diagnostics(
    cfg: ReasonerConfig,
    batch_size: int = 2,
    seq_len: int = 16,
    recurrent_depths: tuple[int, ...] = (1, 2, 4),
) -> dict[str, Any]:
    """Run the Tier-1 architecture-prior diagnostics suite."""

    probes = [
        activation_sanity_probe(cfg, batch_size, seq_len),
        positional_addressing_probe(cfg, batch_size, seq_len),
        identity_signal_probe(cfg, batch_size, seq_len),
        memory_mechanics_probe(cfg, batch_size, seq_len),
        recurrent_consistency_probe(cfg, batch_size, seq_len, recurrent_depths),
        routing_health_probe(cfg, batch_size, seq_len),
    ]
    scores = {probe.name: probe.score for probe in probes}
    aggregate = sum(scores.values()) / max(1, len(scores))
    return {
        "aggregate_prior_score": float(aggregate),
        "scores": scores,
        "probes": [probe.to_dict() for probe in probes],
        "interpretation": "architecture-prior diagnostic only; no gradient updates and no semantic reasoning claim",
    }
