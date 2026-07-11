from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, cast

import torch

from wai_r0.config import ReasonerConfig
from wai_r0.core.runtime import temporary_torch_threads
from wai_r0.eval.algorithmic import (
    DEFAULT_VOCAB_SIZE,
    AlgorithmicBatchStream,
    AlgorithmicTask,
)
from wai_r0.eval.gates import decide_non_compensatory
from wai_r0.eval.metrics import evaluate_sequence_batches
from wai_r0.experiments.manifest import ExperimentManifest, load_experiment_manifest
from wai_r0.experiments.statistics import PairedComparison, compare_paired
from wai_r0.model import ModelOutput, ReasonerCore
from wai_r0.profiler import ProfileResult, profile_model
from wai_r0.reporting.schema import (
    Decision,
    ResearchReport,
    RunIdentity,
    default_hardware_info,
    default_software_info,
    write_report,
)
from wai_r0.training.engine import Trainer, TrainerConfig
from wai_r0.training.schedules import ScheduleName


class ExperimentExecutionError(RuntimeError):
    pass


def _validate_execution_budget(manifest: ExperimentManifest) -> None:
    if manifest.kind == "algorithmic":
        step_limit = manifest.maximum_budget.get("optimizer_steps_per_variant")
        if step_limit is not None:
            requested = int(manifest.execution.get("train_steps", 100))
            if requested > int(step_limit):
                raise ExperimentExecutionError(
                    "algorithmic train_steps exceeds optimizer_steps_per_variant budget: "
                    f"requested={requested}, limit={int(step_limit)}"
                )
    if manifest.kind == "profile":
        run_limit = manifest.maximum_budget.get("measured_runs_per_seed")
        if run_limit is not None:
            requested = int(manifest.execution.get("measured_runs", 5))
            if requested > int(run_limit):
                raise ExperimentExecutionError(
                    "profile measured_runs exceeds measured_runs_per_seed budget: "
                    f"requested={requested}, limit={int(run_limit)}"
                )


def _load_model_config(reference: str, *, base_dir: Path) -> ReasonerConfig:
    path = Path(reference)
    if not path.is_absolute():
        path = base_dir / path
    return ReasonerConfig.from_yaml(path)


def _seeded(config: ReasonerConfig, seed: int) -> ReasonerConfig:
    return ReasonerConfig.from_dict({**config.to_dict(), "seed": seed})


def _matching_failure(
    manifest: ExperimentManifest,
    candidate: ReasonerCore,
    control: ReasonerCore,
) -> str | None:
    candidate_counts = candidate.parameter_counts()
    control_counts = control.parameter_counts()
    if manifest.matching_rule == "parameter_matched":
        if candidate_counts["total"] != control_counts["total"]:
            return (
                "parameter_matched comparison is invalid: "
                f"candidate={candidate_counts['total']}, control={control_counts['total']}"
            )
    elif manifest.matching_rule == "active_parameter_matched":
        candidate_active = candidate_counts["active_per_token_estimate"]
        control_active = control_counts["active_per_token_estimate"]
        if candidate_active != control_active:
            return (
                "active_parameter_matched comparison is invalid: "
                f"candidate={candidate_active}, control={control_active}"
            )
    elif manifest.matching_rule in {"flop_matched", "wall_clock_matched", "memory_matched"}:
        return f"{manifest.matching_rule} execution is not implemented by this runner"
    return None


@torch.inference_mode()
def _cache_equivalence(core: ReasonerCore, *, seed: int, sequence_length: int) -> float:
    generator = torch.Generator(device=core.device_obj.type)
    generator.manual_seed(seed)
    prompt_len = max(2, min(sequence_length, core.cfg.max_seq_len - 1))
    prompt = torch.randint(
        0,
        core.cfg.vocab_size,
        (1, prompt_len),
        generator=generator,
        device=core.device_obj,
    )
    next_token = torch.randint(
        0,
        core.cfg.vocab_size,
        (1, 1),
        generator=generator,
        device=core.device_obj,
    )
    prefill = core.transformer(prompt, use_cache=True, return_dict=True)
    if not isinstance(prefill, ModelOutput) or prefill.past_key_values is None:
        raise ExperimentExecutionError("prefill did not produce a cache")
    cached = core.transformer(
        next_token,
        past_key_values=prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    full = core.transformer(torch.cat((prompt, next_token), dim=1), return_dict=True)
    if not isinstance(cached, ModelOutput) or not isinstance(full, ModelOutput):
        raise ExperimentExecutionError("structured output contract failed")
    return float((cached.logits[:, -1] - full.logits[:, -1]).abs().max().float().cpu())


def _profile_metric(name: str, candidate: ProfileResult, control: ProfileResult) -> float:
    pairs: dict[str, tuple[float, float]] = {
        "kv_cache_reduction": (candidate.kv_cache_bytes, control.kv_cache_bytes),
        "decode_latency_improvement": (
            candidate.decode_latency_ms_median,
            control.decode_latency_ms_median,
        ),
        "prefill_latency_improvement": (
            candidate.prefill_latency_ms_median,
            control.prefill_latency_ms_median,
        ),
        "parameter_memory_reduction": (candidate.parameter_bytes, control.parameter_bytes),
    }
    if name == "peak_memory_reduction":
        if candidate.peak_allocated_bytes is None or control.peak_allocated_bytes is None:
            raise ExperimentExecutionError("peak_memory_reduction requires CUDA measurements")
        candidate_value = float(candidate.peak_allocated_bytes)
        control_value = float(control.peak_allocated_bytes)
    else:
        try:
            candidate_value, control_value = pairs[name]
        except KeyError as exc:
            raise ExperimentExecutionError(f"unsupported profile metric: {name}") from exc
    if control_value == 0:
        raise ExperimentExecutionError(f"control value for {name} is zero")
    return 1.0 - float(candidate_value) / float(control_value)


def _profile_experiment(
    manifest: ExperimentManifest,
    *,
    base_dir: Path,
) -> tuple[dict[str, Any], PairedComparison | None, list[str]]:
    execution = manifest.execution
    candidate_base = _load_model_config(
        str(execution.get("candidate_config", manifest.candidate)), base_dir=base_dir
    )
    control_base = _load_model_config(
        str(execution.get("control_config", manifest.control)), base_dir=base_dir
    )
    batch_size = int(execution.get("batch_size", 1))
    sequence_length = int(execution.get("sequence_length", 32))
    warmup_runs = int(execution.get("warmup_runs", 2))
    measured_runs = int(execution.get("measured_runs", 5))
    equivalence_tolerance = float(execution.get("cache_equivalence_tolerance", 1e-4))
    cpu_threads = (
        int(execution["cpu_threads"]) if execution.get("cpu_threads") is not None else None
    )

    seed_rows: list[dict[str, Any]] = []
    candidate_values: list[float] = []
    control_values: list[float] = []
    correctness_failures: list[str] = []
    for seed in manifest.seeds:
        candidate = ReasonerCore(_seeded(candidate_base, seed))
        control = ReasonerCore(_seeded(control_base, seed))
        matching = _matching_failure(manifest, candidate, control)
        if matching:
            correctness_failures.append(matching)
        candidate_error = _cache_equivalence(candidate, seed=seed, sequence_length=sequence_length)
        control_error = _cache_equivalence(control, seed=seed, sequence_length=sequence_length)
        if candidate_error > equivalence_tolerance:
            correctness_failures.append(
                f"seed {seed}: candidate cache max error {candidate_error:.3g} exceeds "
                f"{equivalence_tolerance:.3g}"
            )
        if control_error > equivalence_tolerance:
            correctness_failures.append(
                f"seed {seed}: control cache max error {control_error:.3g} exceeds "
                f"{equivalence_tolerance:.3g}"
            )
        candidate_profile = profile_model(
            candidate.transformer,
            batch_size=batch_size,
            sequence_length=sequence_length,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            cpu_threads=cpu_threads,
        )
        control_profile = profile_model(
            control.transformer,
            batch_size=batch_size,
            sequence_length=sequence_length,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            cpu_threads=cpu_threads,
        )
        metric = _profile_metric(manifest.primary_metric, candidate_profile, control_profile)
        # The paired comparison receives metric contributions against a zero
        # control baseline so its CI is a CI over the preregistered improvement.
        candidate_values.append(metric)
        control_values.append(0.0)
        seed_rows.append(
            {
                "seed": seed,
                "primary_metric": metric,
                "candidate": candidate_profile.to_dict(),
                "control": control_profile.to_dict(),
                "cache_equivalence_max_error": {
                    "candidate": candidate_error,
                    "control": control_error,
                },
                "parameter_counts": {
                    "candidate": candidate.parameter_counts(),
                    "control": control.parameter_counts(),
                },
            }
        )
    comparison = compare_paired(
        candidate_values,
        control_values,
        higher_is_better=manifest.thresholds.higher_is_better,
        tie_tolerance=manifest.tie_tolerance,
        bootstrap_seed=manifest.seeds[0],
    )
    return {"seeds": seed_rows}, comparison, correctness_failures


def _algorithmic_metric(name: str, metrics: dict[str, Any]) -> float:
    aliases = {
        "id_exact_match": ("in_distribution", "exact_match"),
        "id_token_accuracy": ("in_distribution", "token_accuracy"),
        "ood_exact_match": ("out_of_distribution", "exact_match"),
        "ood_token_accuracy": ("out_of_distribution", "token_accuracy"),
        "negative_final_loss": ("training", "negative_final_loss"),
    }
    try:
        section, key = aliases[name]
    except KeyError as exc:
        raise ExperimentExecutionError(f"unsupported algorithmic metric: {name}") from exc
    return float(metrics[section][key])


def _run_algorithmic_variant(
    config: ReasonerConfig,
    *,
    task: AlgorithmicTask,
    seed: int,
    execution: dict[str, Any],
) -> dict[str, Any]:
    cpu_threads = (
        int(execution["cpu_threads"]) if execution.get("cpu_threads") is not None else None
    )
    requested_threads = cpu_threads if config.device == "cpu" else None
    with temporary_torch_threads(requested_threads):
        return _run_algorithmic_variant_impl(
            config, task=task, seed=seed, execution=execution, cpu_threads=cpu_threads
        )


def _run_algorithmic_variant_impl(
    config: ReasonerConfig,
    *,
    task: AlgorithmicTask,
    seed: int,
    execution: dict[str, Any],
    cpu_threads: int | None,
) -> dict[str, Any]:
    if config.vocab_size < DEFAULT_VOCAB_SIZE:
        raise ExperimentExecutionError(
            f"algorithmic experiments require vocab_size >= {DEFAULT_VOCAB_SIZE}"
        )
    train_lengths = tuple(int(value) for value in execution.get("train_lengths", [8]))
    id_length = int(execution.get("id_length", max(train_lengths)))
    ood_length = int(execution.get("ood_length", max(train_lengths) * 2))
    batch_size = int(execution.get("batch_size", 8))
    train_steps = int(execution.get("train_steps", 100))
    eval_batches = int(execution.get("eval_batches", 8))
    learning_rate = float(execution.get("learning_rate", 3e-4))
    max_length = config.max_seq_len

    model = ReasonerCore(_seeded(config, seed))
    source = AlgorithmicBatchStream(
        task,
        seed=seed + 10_000,
        batch_size=batch_size,
        lengths=train_lengths,
        max_length=max_length,
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            max_steps=train_steps,
            learning_rate=learning_rate,
            weight_decay=float(execution.get("weight_decay", 0.01)),
            gradient_accumulation_steps=int(execution.get("gradient_accumulation_steps", 1)),
            max_gradient_norm=float(execution.get("max_gradient_norm", 1.0)),
            mixed_precision=str(execution.get("mixed_precision", "none")),
            schedule=cast(ScheduleName, str(execution.get("schedule", "cosine"))),
            warmup_steps=int(execution.get("warmup_steps", 0)),
            checkpoint_every=0,
            save_final_checkpoint=False,
            model_mode=str(execution.get("model_mode", "fast")),
            recurrent_steps=(
                int(execution["recurrent_steps"])
                if execution.get("recurrent_steps") is not None
                else None
            ),
            cpu_threads=cpu_threads,
        ),
    )
    training = trainer.train(source)
    if not training.metrics:
        raise ExperimentExecutionError("algorithmic training produced no metrics")
    model_mode = str(execution.get("model_mode", "fast"))
    recurrent_steps = (
        int(execution["recurrent_steps"]) if execution.get("recurrent_steps") is not None else None
    )
    in_distribution = evaluate_sequence_batches(
        model,
        AlgorithmicBatchStream(
            task,
            seed=seed + 20_000,
            batch_size=batch_size,
            lengths=[id_length],
            max_length=max_length,
        ),
        max_batches=eval_batches,
        model_mode=model_mode,
        recurrent_steps=recurrent_steps,
    )
    out_of_distribution = evaluate_sequence_batches(
        model,
        AlgorithmicBatchStream(
            task,
            seed=seed + 30_000,
            batch_size=batch_size,
            lengths=[ood_length],
            max_length=max_length,
        ),
        max_batches=eval_batches,
        model_mode=model_mode,
        recurrent_steps=recurrent_steps,
    )
    final = training.metrics[-1]
    return {
        "training": {
            "steps": training.progress.global_step,
            "target_tokens": training.progress.consumed_tokens,
            "final_loss": final.loss,
            "negative_final_loss": -final.loss,
            "target_tokens_per_second": final.target_tokens_per_second,
            "cpu_threads": cpu_threads if config.device == "cpu" else None,
        },
        "in_distribution": in_distribution.to_dict(),
        "out_of_distribution": out_of_distribution.to_dict(),
        "parameter_counts": model.parameter_counts(),
    }


def _algorithmic_experiment(
    manifest: ExperimentManifest,
    *,
    base_dir: Path,
) -> tuple[dict[str, Any], PairedComparison | None, list[str]]:
    execution = manifest.execution
    task_name = str(execution.get("task", manifest.datasets[0]))
    allowed_tasks = {
        "copy",
        "reverse",
        "parity",
        "modular_addition",
        "sorting",
        "selective_copy",
        "associative_recall",
        "bracket_balance",
        "finite_state_parity",
    }
    if task_name not in allowed_tasks:
        raise ExperimentExecutionError(f"unsupported algorithmic task: {task_name}")
    task = cast(AlgorithmicTask, task_name)
    candidate_base = _load_model_config(
        str(execution.get("candidate_config", manifest.candidate)), base_dir=base_dir
    )
    control_base = _load_model_config(
        str(execution.get("control_config", manifest.control)), base_dir=base_dir
    )
    seed_rows: list[dict[str, Any]] = []
    candidate_values: list[float] = []
    control_values: list[float] = []
    correctness_failures: list[str] = []
    for seed in manifest.seeds:
        candidate_probe = ReasonerCore(_seeded(candidate_base, seed))
        control_probe = ReasonerCore(_seeded(control_base, seed))
        matching = _matching_failure(manifest, candidate_probe, control_probe)
        if matching:
            correctness_failures.append(matching)
        try:
            candidate_execution = {
                **execution,
                "model_mode": execution.get(
                    "candidate_model_mode", execution.get("model_mode", "fast")
                ),
                "recurrent_steps": execution.get(
                    "candidate_recurrent_steps", execution.get("recurrent_steps")
                ),
            }
            control_execution = {
                **execution,
                "model_mode": execution.get(
                    "control_model_mode", execution.get("model_mode", "fast")
                ),
                "recurrent_steps": execution.get(
                    "control_recurrent_steps", execution.get("recurrent_steps")
                ),
            }
            candidate_metrics = _run_algorithmic_variant(
                candidate_base, task=task, seed=seed, execution=candidate_execution
            )
            control_metrics = _run_algorithmic_variant(
                control_base, task=task, seed=seed, execution=control_execution
            )
            candidate_value = _algorithmic_metric(manifest.primary_metric, candidate_metrics)
            control_value = _algorithmic_metric(manifest.primary_metric, control_metrics)
            if not math.isfinite(candidate_value) or not math.isfinite(control_value):
                raise ExperimentExecutionError("non-finite primary metric")
        except (RuntimeError, ValueError) as exc:
            correctness_failures.append(f"seed {seed}: {exc}")
            continue
        candidate_values.append(candidate_value)
        control_values.append(control_value)
        seed_rows.append(
            {
                "seed": seed,
                "candidate": candidate_metrics,
                "control": control_metrics,
                "candidate_primary": candidate_value,
                "control_primary": control_value,
                "difference": candidate_value - control_value,
            }
        )
    comparison = None
    if candidate_values:
        comparison = compare_paired(
            candidate_values,
            control_values,
            higher_is_better=manifest.thresholds.higher_is_better,
            tie_tolerance=manifest.tie_tolerance,
            bootstrap_seed=manifest.seeds[0],
        )
    return {"task": task, "seeds": seed_rows}, comparison, correctness_failures


def _external_metrics_experiment(
    manifest: ExperimentManifest,
) -> tuple[dict[str, Any], PairedComparison | None, list[str]]:
    candidate = manifest.execution.get("candidate_values")
    control = manifest.execution.get("control_values")
    if not isinstance(candidate, list) or not isinstance(control, list):
        raise ExperimentExecutionError(
            "external_metrics execution requires candidate_values and control_values lists"
        )
    comparison = compare_paired(
        [float(value) for value in candidate],
        [float(value) for value in control],
        higher_is_better=manifest.thresholds.higher_is_better,
        tie_tolerance=manifest.tie_tolerance,
        bootstrap_seed=manifest.seeds[0],
    )
    return {"candidate_values": candidate, "control_values": control}, comparison, []


def run_experiment(
    manifest_or_path: ExperimentManifest | str | Path,
    *,
    output: str | Path | None = None,
    repository: str | Path | None = None,
) -> ResearchReport:
    manifest_path: Path | None = None
    if isinstance(manifest_or_path, ExperimentManifest):
        manifest = manifest_or_path
        base_dir = Path.cwd()
    else:
        manifest_path = Path(manifest_or_path).resolve()
        manifest = load_experiment_manifest(manifest_path)
        base_dir = manifest_path.parent
    manifest.validate()
    _validate_execution_budget(manifest)
    started = time.perf_counter()
    if manifest.kind == "profile":
        raw, comparison, failures = _profile_experiment(manifest, base_dir=base_dir)
    elif manifest.kind == "algorithmic":
        raw, comparison, failures = _algorithmic_experiment(manifest, base_dir=base_dir)
    else:
        raw, comparison, failures = _external_metrics_experiment(manifest)
    elapsed = time.perf_counter() - started
    wall_clock_limit = manifest.maximum_budget.get("wall_clock_seconds")
    if wall_clock_limit is not None and elapsed > float(wall_clock_limit):
        failures.append(
            f"wall-clock budget exceeded: observed {elapsed:.3f}s, limit {float(wall_clock_limit):.3f}s"
        )

    primary_value = comparison.oriented_mean_difference if comparison is not None else None
    gates, decision = decide_non_compensatory(
        manifest,
        primary_value=primary_value,
        successful_seeds=comparison.count if comparison is not None else 0,
        correctness_failures=failures,
        robustness_results=raw.get("robustness"),
    )
    metrics: dict[str, Any] = {
        "experiment_id": manifest.id,
        "primary_metric": manifest.primary_metric,
        "elapsed_seconds": elapsed,
        "raw": raw,
        "paired_comparison": comparison.to_dict() if comparison is not None else None,
    }
    identity = RunIdentity.create(
        command=["wai-r0", "experiment", "run", str(manifest_path or manifest.id)],
        config=manifest.to_dict(),
        experiment_hash=manifest.manifest_hash,
        repository=repository,
    )
    limitations = list(manifest.known_confounds) or [
        "Local small-model evidence does not establish scale transfer."
    ]
    report = ResearchReport(
        identity=identity,
        evidence_class=manifest.evidence_class,
        resolved_config=manifest.to_dict(),
        metrics=metrics,
        gates=gates,
        decision=cast(Decision, decision),
        limitations=limitations,
        hardware=default_hardware_info(),
        software=default_software_info(),
        failures=failures,
        provenance={
            "manifest_hash": manifest.manifest_hash,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "final_evaluation": manifest.final_evaluation,
            "matching_rule": manifest.matching_rule,
        },
        artifacts={"manifest": str(manifest_path)} if manifest_path else {},
    )
    report.validate()
    if output is not None:
        write_report(output, report)
    return report


__all__ = ["ExperimentExecutionError", "run_experiment"]
