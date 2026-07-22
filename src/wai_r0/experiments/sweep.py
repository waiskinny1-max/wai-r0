from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wai_r0.core.reproducibility import atomic_write_json, canonical_hash
from wai_r0.experiments.manifest import ExperimentManifest
from wai_r0.experiments.runner import run_experiment


@dataclass(frozen=True, slots=True)
class SweepSpec:
    id: str
    base_manifest: str
    grid: dict[str, list[Any]]
    maximum_trials: int = 64

    def validate(self) -> None:
        if not self.id.strip() or any(character.isspace() for character in self.id):
            raise ValueError("sweep id must be non-empty and contain no whitespace")
        if not self.base_manifest.strip():
            raise ValueError("base_manifest cannot be empty")
        if not self.grid:
            raise ValueError("sweep grid cannot be empty")
        if self.maximum_trials < 1:
            raise ValueError("maximum_trials must be positive")
        total = 1
        for path, values in self.grid.items():
            if not path.strip() or path.startswith(".") or path.endswith("."):
                raise ValueError(f"invalid sweep parameter path: {path!r}")
            if not values:
                raise ValueError(f"sweep parameter {path!r} has no values")
            total *= len(values)
        if total > self.maximum_trials:
            raise ValueError(
                f"sweep expands to {total} trials, above maximum_trials={self.maximum_trials}"
            )

    @property
    def trial_count(self) -> int:
        total = 1
        for values in self.grid.values():
            total *= len(values)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "base_manifest": self.base_manifest,
            "grid": {key: list(values) for key, values in sorted(self.grid.items())},
            "maximum_trials": self.maximum_trials,
        }


@dataclass(frozen=True, slots=True)
class SweepTrial:
    index: int
    trial_id: str
    parameters: dict[str, Any]
    manifest: ExperimentManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "trial_id": self.trial_id,
            "parameters": dict(sorted(self.parameters.items())),
            "manifest_hash": self.manifest.manifest_hash,
            "manifest": self.manifest.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SweepPlan:
    spec: SweepSpec
    spec_path: str
    base_manifest_path: str
    trials: tuple[SweepTrial, ...]

    @property
    def plan_hash(self) -> str:
        return canonical_hash(
            {
                "spec": self.spec.to_dict(),
                "base_manifest_path": self.base_manifest_path,
                "trials": [trial.to_dict() for trial in self.trials],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "spec_path": self.spec_path,
            "base_manifest_path": self.base_manifest_path,
            "trial_count": len(self.trials),
            "plan_hash": self.plan_hash,
            "trials": [trial.to_dict() for trial in self.trials],
        }


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            raise ValueError(f"sweep path does not resolve to a mapping: {dotted_path!r}")
        cursor = existing
    leaf = parts[-1]
    if leaf not in cursor:
        raise ValueError(f"sweep path does not exist in base manifest: {dotted_path!r}")
    cursor[leaf] = copy.deepcopy(value)


def _resolve_execution_paths(payload: dict[str, Any], *, base_dir: Path) -> None:
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        return
    for key, value in tuple(execution.items()):
        if not key.endswith("_config") or not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            execution[key] = str((base_dir / path).resolve())


def load_sweep_spec(path: str | Path) -> SweepSpec:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sweep specification must be a mapping")
    allowed = {"id", "base_manifest", "grid", "maximum_trials"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown sweep fields: {', '.join(unknown)}")
    raw_grid = payload.get("grid")
    if not isinstance(raw_grid, Mapping):
        raise ValueError("sweep grid must be a mapping")
    grid: dict[str, list[Any]] = {}
    for key, values in raw_grid.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"sweep parameter {key!r} must be a list")
        grid[str(key)] = list(values)
    spec = SweepSpec(
        id=str(payload.get("id", "")),
        base_manifest=str(payload.get("base_manifest", "")),
        grid=grid,
        maximum_trials=int(payload.get("maximum_trials", 64)),
    )
    spec.validate()
    return spec


def build_sweep_plan(path: str | Path) -> SweepPlan:
    source = Path(path).resolve()
    spec = load_sweep_spec(source)
    base_path = Path(spec.base_manifest)
    if not base_path.is_absolute():
        base_path = (source.parent / base_path).resolve()
    raw_base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw_base, dict):
        raise ValueError("base experiment manifest must be a mapping")

    parameter_names = sorted(spec.grid)
    trials: list[SweepTrial] = []
    combinations = itertools.product(*(spec.grid[name] for name in parameter_names))
    for index, combination in enumerate(combinations):
        parameters = dict(zip(parameter_names, combination, strict=True))
        payload = copy.deepcopy(raw_base)
        for dotted_path, value in parameters.items():
            _set_nested(payload, dotted_path, value)
        _resolve_execution_paths(payload, base_dir=base_path.parent)
        suffix = canonical_hash(parameters)[:10]
        payload["id"] = f"{raw_base.get('id', spec.id)}--{suffix}"
        manifest = ExperimentManifest.from_dict(payload)
        trials.append(
            SweepTrial(
                index=index,
                trial_id=f"{spec.id}-{index:04d}-{suffix}",
                parameters=parameters,
                manifest=manifest,
            )
        )
    if len(trials) != spec.trial_count:
        raise RuntimeError("sweep expansion produced an unexpected trial count")
    return SweepPlan(
        spec=spec,
        spec_path=str(source),
        base_manifest_path=str(base_path),
        trials=tuple(trials),
    )


def write_sweep_plan(plan: SweepPlan, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    trials_dir = root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    for trial in plan.trials:
        trial_path = trials_dir / f"{trial.trial_id}.yaml"
        trial_path.write_text(
            yaml.safe_dump(trial.manifest.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
    return atomic_write_json(root / "plan.json", plan.to_dict())


def run_sweep(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    repository: str | Path | None = None,
    maximum_trials: int | None = None,
    stop_on_failure: bool = False,
) -> dict[str, Any]:
    plan = build_sweep_plan(spec_path)
    root = Path(output_dir)
    plan_path = write_sweep_plan(plan, root)
    selected = plan.trials if maximum_trials is None else plan.trials[:maximum_trials]
    if maximum_trials is not None and maximum_trials < 1:
        raise ValueError("maximum_trials must be positive")

    outcomes: list[dict[str, Any]] = []
    for trial in selected:
        manifest_path = root / "trials" / f"{trial.trial_id}.yaml"
        report_path = root / "reports" / f"{trial.trial_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            report = run_experiment(manifest_path, output=report_path, repository=repository)
        except (OSError, RuntimeError, ValueError) as exc:
            outcomes.append(
                {
                    "trial_id": trial.trial_id,
                    "status": "failed",
                    "parameters": trial.parameters,
                    "error": str(exc),
                }
            )
            if stop_on_failure:
                break
        else:
            outcomes.append(
                {
                    "trial_id": trial.trial_id,
                    "status": "completed",
                    "parameters": trial.parameters,
                    "run_id": report.identity.run_id,
                    "decision": report.decision,
                    "report": str(report_path),
                }
            )

    summary = {
        "sweep_id": plan.spec.id,
        "plan_hash": plan.plan_hash,
        "plan": str(plan_path),
        "planned_trials": len(plan.trials),
        "executed_trials": len(outcomes),
        "completed_trials": sum(item["status"] == "completed" for item in outcomes),
        "failed_trials": sum(item["status"] == "failed" for item in outcomes),
        "outcomes": outcomes,
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


__all__ = [
    "SweepPlan",
    "SweepSpec",
    "SweepTrial",
    "build_sweep_plan",
    "load_sweep_spec",
    "run_sweep",
    "write_sweep_plan",
]
