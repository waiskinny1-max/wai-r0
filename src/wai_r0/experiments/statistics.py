from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SampleSummary:
    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    median: float
    minimum: float
    maximum: float
    confidence_interval_95: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedComparison:
    count: int
    candidate: SampleSummary
    control: SampleSummary
    mean_difference: float
    oriented_mean_difference: float
    relative_improvement: float | None
    difference_confidence_interval_95: tuple[float, float]
    bootstrap_confidence_interval_95: tuple[float, float]
    paired_effect_size_dz: float | None
    exact_sign_test_p_value: float
    wins: int
    ties: int
    losses: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_T_95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _finite_values(values: Sequence[float]) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError("at least one value is required")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("all values must be finite")
    return converted


def _confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        return mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(count)
    critical = _T_95[count - 2] if count <= 31 else 1.96
    margin = critical * standard_error
    return mean - margin, mean + margin


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 4000, seed: int = 0
) -> tuple[float, float]:
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = []
    count = len(values)
    for _ in range(samples):
        means.append(statistics.fmean(values[rng.randrange(count)] for _ in range(count)))
    means.sort()
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return lower, upper


def _two_sided_sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = float(sum(math.comb(trials, index) for index in range(extreme + 1))) / float(2**trials)
    return min(1.0, 2.0 * tail)


def summarize_samples(values: Sequence[float]) -> SampleSummary:
    samples = _finite_values(values)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return SampleSummary(
        count=len(samples),
        mean=statistics.fmean(samples),
        standard_deviation=standard_deviation,
        standard_error=standard_deviation / math.sqrt(len(samples)),
        median=statistics.median(samples),
        minimum=min(samples),
        maximum=max(samples),
        confidence_interval_95=_confidence_interval(samples),
    )


def compare_paired(
    candidate_values: Sequence[float],
    control_values: Sequence[float],
    *,
    higher_is_better: bool = True,
    tie_tolerance: float = 0.0,
    bootstrap_samples: int = 4000,
    bootstrap_seed: int = 0,
) -> PairedComparison:
    candidate = _finite_values(candidate_values)
    control = _finite_values(control_values)
    if len(candidate) != len(control):
        raise ValueError("candidate and control must have the same number of paired values")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance cannot be negative")

    raw_differences = [left - right for left, right in zip(candidate, control, strict=True)]
    oriented = raw_differences if higher_is_better else [-value for value in raw_differences]
    wins = sum(value > tie_tolerance for value in oriented)
    losses = sum(value < -tie_tolerance for value in oriented)
    ties = len(oriented) - wins - losses
    difference_sd = statistics.stdev(raw_differences) if len(raw_differences) > 1 else 0.0
    effect_size = statistics.fmean(oriented) / difference_sd if difference_sd > 0 else None
    control_mean = statistics.fmean(control)
    oriented_mean = statistics.fmean(oriented)
    relative = oriented_mean / abs(control_mean) if control_mean != 0 else None
    return PairedComparison(
        count=len(candidate),
        candidate=summarize_samples(candidate),
        control=summarize_samples(control),
        mean_difference=statistics.fmean(raw_differences),
        oriented_mean_difference=oriented_mean,
        relative_improvement=relative,
        difference_confidence_interval_95=_confidence_interval(raw_differences),
        bootstrap_confidence_interval_95=_bootstrap_mean_interval(
            oriented, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        paired_effect_size_dz=effect_size,
        exact_sign_test_p_value=_two_sided_sign_test(wins, losses),
        wins=wins,
        ties=ties,
        losses=losses,
    )
