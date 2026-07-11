from __future__ import annotations

from collections.abc import Mapping, Sequence

from wai_r0.experiments.manifest import ExperimentManifest
from wai_r0.reporting.schema import GateResult


def decide_non_compensatory(
    manifest: ExperimentManifest,
    *,
    primary_value: float | None,
    successful_seeds: int,
    correctness_failures: Sequence[str] = (),
    robustness_results: Mapping[str, bool] | None = None,
) -> tuple[list[GateResult], str]:
    gates: list[GateResult] = []
    correctness_ok = not correctness_failures
    gates.append(
        GateResult(
            name="correctness",
            status="pass" if correctness_ok else "fail",
            explanation=(
                "All candidate/control runs produced finite, structurally valid outputs."
                if correctness_ok
                else "; ".join(correctness_failures)
            ),
        )
    )
    seed_ok = successful_seeds >= manifest.required_successful_seeds
    gates.append(
        GateResult(
            name="successful_seed_count",
            status="pass" if seed_ok else "fail",
            explanation=(
                f"{successful_seeds} successful paired seeds; "
                f"{manifest.required_successful_seeds} required."
            ),
            observed=float(successful_seeds),
            threshold=float(manifest.required_successful_seeds),
        )
    )

    if primary_value is None:
        gates.append(
            GateResult(
                name="primary_metric",
                status="not_run",
                explanation="No finite paired primary metric was available.",
                metric=manifest.primary_metric,
            )
        )
        threshold_decision = "re_test"
    else:
        threshold_decision = manifest.thresholds.decide(primary_value)
        primary_pass = threshold_decision == "keep"
        threshold = (
            manifest.thresholds.keep
            if primary_pass
            else manifest.thresholds.kill
            if threshold_decision == "kill"
            else manifest.thresholds.keep
        )
        gates.append(
            GateResult(
                name="primary_metric",
                status="pass"
                if primary_pass
                else "fail"
                if threshold_decision == "kill"
                else "inconclusive",
                explanation=(f"Preregistered threshold decision: {threshold_decision}."),
                metric=manifest.primary_metric,
                observed=primary_value,
                threshold=threshold,
            )
        )

    robustness = dict(robustness_results or {})
    for axis in manifest.robustness_axes:
        observed = robustness.get(axis)
        gates.append(
            GateResult(
                name=f"robustness:{axis}",
                status="pass" if observed is True else "fail" if observed is False else "not_run",
                explanation=(
                    "Robustness condition passed."
                    if observed is True
                    else "Robustness condition failed."
                    if observed is False
                    else "Robustness axis was preregistered but not executed."
                ),
            )
        )

    hard_failure = any(gate.status == "fail" for gate in gates if gate.name != "primary_metric")
    missing_robustness = any(
        gate.status != "pass" for gate in gates if gate.name.startswith("robustness:")
    )
    if hard_failure or threshold_decision == "kill":
        decision = "kill"
    elif threshold_decision == "keep" and not missing_robustness:
        decision = "keep"
    else:
        decision = "re_test"
    return gates, decision


__all__ = ["decide_non_compensatory"]
