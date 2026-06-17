from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from wai_r0.config import ReasonerConfig
from wai_r0.eval.leakage_guard import LeakageGuard
from wai_r0.eval.prior_diagnostics import run_prior_diagnostics
from wai_r0.model import ReasonerCore, set_seed
from wai_r0.report import BenchmarkReport, recommend
from wai_r0.symbolic import ProgramSearch, load_task
from wai_r0.training.probes import run_tiny_probe


def zero_neural(cfg: ReasonerConfig, batch_size: int = 1, seq_len: int = 16) -> BenchmarkReport:
    set_seed(cfg.seed)
    core = ReasonerCore(cfg)
    tokens = torch.randint(
        0,
        cfg.vocab_size,
        (batch_size, min(seq_len, cfg.max_seq_len)),
        device=core.device_obj,
    )
    logits = core(tokens, mode="think" if cfg.recurrent_depth > 1 else "fast")
    loss = logits.float().mean()
    loss.backward()
    grads = [p.grad.detach().float().norm().item() for p in core.parameters() if p.grad is not None]
    finite = bool(torch.isfinite(logits).all().item())
    fgrads = all(torch.isfinite(torch.tensor(g)).item() for g in grads)
    insp = core.inspect_activations(tokens)
    moe = insp["diagnostics"].get("moe", [])
    collapsed = any(m.get("collapse_warning", False) for m in moe)
    rec = insp.get("recurrent")
    recurrent_ok = rec is None or max(rec.get("norm_by_step", [0.0])) < 1e4
    score = 0.4 * finite + 0.3 * fgrads + 0.2 * (not collapsed) + 0.1 * recurrent_ok
    return BenchmarkReport(
        "zero_neural",
        "zero-training neural diagnostic",
        cfg.seed,
        cfg.device,
        cfg.dtype,
        cfg.to_dict(),
        raw_metrics={
            "finite_logits": finite,
            "finite_gradients": fgrads,
            "grad_norm_mean": sum(grads) / max(1, len(grads)),
            "inspection": insp,
            "r0_stability_score": float(score),
        },
        summary="Random-weight numerical diagnostic completed. This is not an intelligence result.",
        limitations=[
            "Random weights do not reason.",
            "Single local run; use multi-seed ablation before any scale decision.",
        ],
        recommendation=recommend(score if finite and fgrads else 0.0),
    )


def architecture_priors(
    cfg: ReasonerConfig,
    batch_size: int = 2,
    seq_len: int = 16,
    recurrent_depths: tuple[int, ...] = (1, 2, 4),
) -> BenchmarkReport:
    raw = run_prior_diagnostics(cfg, batch_size=batch_size, seq_len=seq_len, recurrent_depths=recurrent_depths)
    score = float(raw["aggregate_prior_score"])
    return BenchmarkReport(
        "architecture_priors",
        "architecture-prior diagnostic",
        cfg.seed,
        cfg.device,
        cfg.dtype,
        cfg.to_dict(),
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "recurrent_depths": list(recurrent_depths),
        },
        raw,
        "Tier-1 no-gradient architecture-prior diagnostics completed. This tests mechanics, not reasoning.",
        [
            "No gradient updates are performed.",
            "Identity, position, recurrence, routing, and memory probes are proxies only.",
            "A high prior score does not prove language understanding or frontier reasoning.",
        ],
        recommend(score),
    )


def _attn_cfg(cfg: ReasonerConfig, attn: str) -> ReasonerConfig:
    n_kv = cfg.n_heads if attn == "mha" else (max(1, cfg.n_heads // 2) if attn in {"gqa", "mla_lite"} else cfg.n_kv_heads)
    return ReasonerConfig.from_dict({**cfg.to_dict(), "attention_type": attn, "n_kv_heads": n_kv})


def memory(
    cfg: ReasonerConfig,
    baseline: str,
    candidate: str,
    seq_lens: list[int],
    batch_size: int = 1,
) -> BenchmarkReport:
    base = ReasonerCore(_attn_cfg(cfg, baseline))
    cand = ReasonerCore(_attn_cfg(cfg, candidate))
    rows = []
    for seq_len in seq_lens:
        base_bytes = base.estimate_memory_cost(seq_len, batch_size)["kv_cache_bytes"]
        cand_bytes = cand.estimate_memory_cost(seq_len, batch_size)["kv_cache_bytes"]
        rows.append(
            {
                "seq_len": seq_len,
                "baseline_bytes": base_bytes,
                "candidate_bytes": cand_bytes,
                "candidate_over_baseline": cand_bytes / max(1, base_bytes),
                "saved_bytes": base_bytes - cand_bytes,
            }
        )
    ratio = sum(r["candidate_over_baseline"] for r in rows) / len(rows)
    return BenchmarkReport(
        "memory",
        "architecture-prior diagnostic",
        cfg.seed,
        cfg.device,
        cfg.dtype,
        cfg.to_dict(),
        {"baseline": baseline, "candidate": candidate, "seq_lens": seq_lens},
        {"rows": rows, "average_candidate_over_baseline": ratio},
        "KV-cache memory estimate. Not a reasoning benchmark.",
        ["Static estimate; profile target hardware.", "MLA-lite is not DeepSeek MLA."],
        "TINY-TRAIN ONLY — promising but unproven." if ratio < 1 else "DO NOT TRAIN — architecture has no useful signal.",
    )


def symbolic(
    tasks: str | Path,
    budget_s: float = 10.0,
    max_depth: int = 2,
    leakage_manifest: str | Path | None = None,
    split: str = "dev",
    register_leakage: bool = False,
) -> BenchmarkReport:
    paths = sorted(Path(tasks).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no .json tasks found in {tasks}")

    leakage_summary: dict[str, Any] | None = None
    if leakage_manifest is not None:
        guard = LeakageGuard(leakage_manifest)
        findings = guard.scan_directory(tasks, split=split, register=register_leakage)
        leakage_summary = guard.summary(findings)

    search = ProgramSearch(max_depth=max_depth)
    results = []
    solved = 0
    for path in paths:
        task = load_task(path)
        result = search.solve(task, budget_s)
        solved += int(result.solved)
        results.append({"task_id": task.task_id, **result.to_dict()})
    pass1 = solved / len(paths)
    raw = {"tasks": results, "pass_at_1": pass1}
    if leakage_summary is not None:
        raw["leakage"] = leakage_summary
    limitations = ["Small DSL and demo/generated tasks.", "Avoid public-eval leakage."]
    if leakage_summary and leakage_summary["has_cross_split_duplicates"]:
        limitations.append("Leakage guard detected cross-split duplicate task content.")

    return BenchmarkReport(
        "symbolic_arc",
        "zero-training symbolic solver result",
        0,
        "cpu",
        "n/a",
        benchmark_config={
            "tasks": str(tasks),
            "budget_s": budget_s,
            "max_depth": max_depth,
            "split": split,
            "leakage_manifest": str(leakage_manifest) if leakage_manifest else None,
        },
        raw_metrics=raw,
        summary="Explicit symbolic program search. Not neural reasoning.",
        limitations=limitations,
        recommendation="TINY-TRAIN ONLY — promising but unproven." if pass1 > 0 else "DO NOT TRAIN — architecture has no useful signal.",
    )


def tiny_train(
    cfg: ReasonerConfig,
    task: str = "copy",
    examples: int = 32,
    batch_size: int = 4,
    train_len: int = 8,
    eval_lens: tuple[int, ...] = (8, 16),
) -> BenchmarkReport:
    result = run_tiny_probe(
        cfg,
        task=task,
        examples=examples,
        batch_size=batch_size,
        train_len=train_len,
        eval_lens=eval_lens,
    )
    metrics = result.to_dict()
    return BenchmarkReport(
        "tiny_train",
        "tiny-training architecture probe",
        cfg.seed,
        cfg.device,
        cfg.dtype,
        cfg.to_dict(),
        {"task": task, "examples": examples, "train_len": train_len, "eval_lens": list(eval_lens)},
        metrics,
        "Tiny supervised algorithmic probe with in-length and longer-length evaluation. Not pretrained reasoning.",
        [
            "Tiny smoke budget only.",
            "Eval accuracy is unstable at very low budgets; compare across seeds and baselines.",
        ],
        "TINY-TRAIN ONLY — promising but unproven." if metrics["loss_delta"] > 0 else "DO NOT TRAIN — architecture has no useful signal.",
    )


def _variant_config(base: ReasonerConfig, variant: dict[str, Any], seed: int) -> ReasonerConfig | None:
    attention = variant.get("attention", "mha")
    if attention in {None, "none"}:
        return None
    n_kv = base.n_heads if attention == "mha" else max(1, base.n_heads // 2)
    return ReasonerConfig.from_dict(
        {
            **base.to_dict(),
            "seed": seed,
            "attention_type": attention,
            "n_kv_heads": n_kv,
            "use_moe": bool(variant.get("moe", False)),
            "recurrent_depth": 2 if variant.get("recurrent", False) else 1,
        }
    )


def _average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _variant_decision(seed_rows: list[dict[str, Any]], symbolic_pass: float | None) -> str:
    finite_rows = [r for r in seed_rows if r.get("finite_logits") and r.get("finite_gradients")]
    if seed_rows and len(finite_rows) != len(seed_rows):
        return "kill"
    if symbolic_pass is not None and symbolic_pass <= 0 and not seed_rows:
        return "kill"
    stability = _average([float(r.get("r0_stability_score", 0.0)) for r in seed_rows]) if seed_rows else 0.0
    loss_delta = _average([float(r.get("tiny_loss_delta", 0.0)) for r in seed_rows]) if seed_rows else 0.0
    if stability >= 0.9 and (loss_delta > 0 or (symbolic_pass is not None and symbolic_pass > 0)):
        return "keep"
    return "re-test"


def ablate(
    cfg: ReasonerConfig,
    matrix_path: str | Path,
    seeds: list[int] | None = None,
    tasks: str | Path | None = "examples/tasks",
    tiny_examples: int = 8,
) -> BenchmarkReport:
    raw = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "variants" not in raw:
        raise ValueError("ablation matrix must contain a variants list")
    seeds = list(seeds or raw.get("seeds") or [cfg.seed])
    rows = []

    for variant in raw["variants"]:
        seed_rows: list[dict[str, Any]] = []
        symbolic_pass: float | None = None
        if variant.get("symbolic_shell", False) and tasks:
            sr = symbolic(tasks, budget_s=float(raw.get("symbolic_budget_s", 3.0)), max_depth=int(raw.get("symbolic_max_depth", 2)))
            symbolic_pass = float(sr.raw_metrics.get("pass_at_1", 0.0))

        for seed in seeds:
            vc = _variant_config(cfg, variant, seed)
            if vc is None:
                continue
            zero = zero_neural(vc, batch_size=1, seq_len=min(8, vc.max_seq_len))
            tiny = tiny_train(vc, examples=tiny_examples, train_len=min(8, vc.max_seq_len), eval_lens=(min(8, vc.max_seq_len), min(16, vc.max_seq_len)))
            seed_rows.append(
                {
                    "seed": seed,
                    "finite_logits": zero.raw_metrics["finite_logits"],
                    "finite_gradients": zero.raw_metrics["finite_gradients"],
                    "r0_stability_score": zero.raw_metrics["r0_stability_score"],
                    "tiny_loss_delta": tiny.raw_metrics["loss_delta"],
                    "tiny_eval_token_accuracy": tiny.raw_metrics["eval_token_accuracy"],
                }
            )

        rows.append(
            {
                "variant": variant["name"],
                "attention": variant.get("attention"),
                "moe": bool(variant.get("moe", False)),
                "recurrent": bool(variant.get("recurrent", False)),
                "symbolic_shell": bool(variant.get("symbolic_shell", False)),
                "symbolic_pass_at_1": symbolic_pass,
                "seeds": seed_rows,
                "decision": _variant_decision(seed_rows, symbolic_pass),
            }
        )

    decisions = {"keep": [], "kill": [], "re-test": []}
    for row in rows:
        decisions[row["decision"]].append(row["variant"])
    stable_count = sum(
        all(seed.get("finite_logits") and seed.get("finite_gradients") for seed in row["seeds"])
        for row in rows
        if row["seeds"]
    )
    return BenchmarkReport(
        "ablation",
        "mixed architecture diagnostic",
        cfg.seed,
        cfg.device,
        cfg.dtype,
        cfg.to_dict(),
        {"matrix": str(matrix_path), "seeds": seeds, "tasks": str(tasks) if tasks else None, "tiny_examples": tiny_examples},
        {"variants": rows, "decisions": decisions, "stable_neural_variant_count": stable_count, "total": len(rows)},
        "Multi-seed ablation over neural architecture and symbolic-shell variants.",
        [
            "Still a small CPU-safe diagnostic.",
            "Decisions are keep/kill/re-test hints, not proof of future scale behavior.",
        ],
        "SCALE TO SMALL — worth 150M–350M experiment." if decisions["keep"] else "TINY-TRAIN ONLY — promising but unproven.",
    )
