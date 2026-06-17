from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ScoreBand = Literal["kill", "retest", "keep"]


@dataclass(frozen=True)
class R0Scorecard:
    stability: float
    memory: float
    symbolic: float
    tiny_train: float
    leakage_penalty: float = 0.0

    @property
    def total(self) -> float:
        score = (
            0.35 * self.stability
            + 0.20 * self.memory
            + 0.20 * self.symbolic
            + 0.25 * self.tiny_train
            - self.leakage_penalty
        )
        return max(0.0, min(1.0, score))

    @property
    def band(self) -> ScoreBand:
        if self.total < 0.35:
            return "kill"
        if self.total < 0.65:
            return "retest"
        return "keep"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total"] = self.total
        data["band"] = self.band
        return data


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_from_report(report: dict[str, Any]) -> R0Scorecard:
    metrics = report.get("raw_metrics", {}) if isinstance(report, dict) else {}
    stability = _as_float(metrics.get("r0_stability_score"), 0.0)
    memory = 1.0 - min(1.0, _as_float(metrics.get("average_candidate_over_baseline"), 1.0))
    symbolic = _as_float(metrics.get("pass_at_1"), 0.0)
    tiny = 1.0 if _as_float(metrics.get("loss_delta"), 0.0) > 0 else 0.0
    leakage = 0.25 if metrics.get("leakage", {}).get("has_cross_split_duplicates") else 0.0
    return R0Scorecard(stability=stability, memory=memory, symbolic=symbolic, tiny_train=tiny, leakage_penalty=leakage)
