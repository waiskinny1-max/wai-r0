from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from wai_r0.core.reproducibility import canonical_hash

MatchingRule = Literal[
    "parameter_matched",
    "active_parameter_matched",
    "flop_matched",
    "wall_clock_matched",
    "memory_matched",
    "token_matched",
]
EvidenceClass = Literal[
    "numerical_diagnostic",
    "architecture_prior",
    "learned_algorithmic",
    "learned_language",
    "symbolic_solver",
    "hybrid_system",
    "systems_performance",
]
ExperimentKind = Literal["profile", "algorithmic", "external_metrics"]


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    keep: float
    kill: float
    higher_is_better: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.keep) or not math.isfinite(self.kill):
            raise ValueError("decision thresholds must be finite")
        if self.higher_is_better and self.kill > self.keep:
            raise ValueError("kill threshold cannot exceed keep threshold")
        if not self.higher_is_better and self.kill < self.keep:
            raise ValueError("kill threshold cannot be below keep threshold")

    def decide(self, value: float) -> Literal["keep", "kill", "re_test"]:
        if not math.isfinite(value):
            raise ValueError("decision value must be finite")
        if self.higher_is_better:
            if value >= self.keep:
                return "keep"
            if value <= self.kill:
                return "kill"
        else:
            if value <= self.keep:
                return "keep"
            if value >= self.kill:
                return "kill"
        return "re_test"


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    id: str
    hypothesis: str
    candidate: str
    control: str
    matching_rule: MatchingRule
    evidence_class: EvidenceClass
    datasets: list[str]
    seeds: list[int]
    primary_metric: str
    thresholds: DecisionThresholds
    kind: ExperimentKind = "external_metrics"
    secondary_metrics: list[str] = field(default_factory=list)
    failure_metrics: list[str] = field(default_factory=list)
    maximum_budget: dict[str, float | int] = field(default_factory=dict)
    known_confounds: list[str] = field(default_factory=list)
    robustness_axes: list[str] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    minimum_successful_seeds: int | None = None
    tie_tolerance: float = 0.0
    final_evaluation: bool = False

    def validate(self) -> None:
        if not self.id.strip() or any(character.isspace() for character in self.id):
            raise ValueError("experiment id must be non-empty and contain no whitespace")
        for name in ("hypothesis", "candidate", "control", "primary_metric"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.candidate == self.control:
            raise ValueError("candidate and control must be different")
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds cannot be negative")
        if any(
            value < 0 or not math.isfinite(float(value)) for value in self.maximum_budget.values()
        ):
            raise ValueError("maximum budget values must be finite and non-negative")
        if (
            self.minimum_successful_seeds is not None
            and not 1 <= self.minimum_successful_seeds <= len(self.seeds)
        ):
            raise ValueError("minimum_successful_seeds is outside the seed count")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance cannot be negative")
        if self.kind == "profile" and self.evidence_class != "systems_performance":
            raise ValueError("profile experiments must use systems_performance evidence")
        if self.kind == "algorithmic" and self.evidence_class != "learned_algorithmic":
            raise ValueError("algorithmic experiments must use learned_algorithmic evidence")
        self.thresholds.validate()

    @property
    def required_successful_seeds(self) -> int:
        return self.minimum_successful_seeds or len(self.seeds)

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentManifest:
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown experiment fields: {', '.join(unknown)}")
        threshold_payload = payload.get("thresholds")
        if not isinstance(threshold_payload, dict):
            raise ValueError("thresholds must be a mapping")
        manifest = cls(
            **{
                **payload,
                "thresholds": DecisionThresholds(**threshold_payload),
            }
        )
        manifest.validate()
        return manifest


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment manifest must be a mapping")
    return ExperimentManifest.from_dict(payload)
