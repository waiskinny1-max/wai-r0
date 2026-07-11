from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

from wai_r0.core.reproducibility import atomic_write_json, canonical_hash
from wai_r0.version import __version__

REPORT_SCHEMA_VERSION = "1.1"
GateStatus = Literal["pass", "fail", "inconclusive", "not_run"]
Decision = Literal["keep", "kill", "re_test", "not_decided"]
EvidenceClass = Literal[
    "numerical_diagnostic",
    "architecture_prior",
    "learned_algorithmic",
    "learned_language",
    "symbolic_solver",
    "hybrid_system",
    "systems_performance",
]


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    created_at_utc: str
    wai_r0_version: str
    command: list[str]
    config_hash: str
    experiment_hash: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None

    @classmethod
    def create(
        cls,
        *,
        command: list[str],
        config: dict[str, Any],
        experiment_hash: str | None = None,
        repository: str | Path | None = None,
    ) -> RunIdentity:
        config_hash = canonical_hash(config)
        timestamp = datetime.now(timezone.utc).isoformat()
        commit, dirty = _git_state(repository)
        run_material = {
            "command": command,
            "config_hash": config_hash,
            "experiment_hash": experiment_hash,
            "git_commit": commit,
        }
        return cls(
            run_id=canonical_hash(run_material)[:16],
            created_at_utc=timestamp,
            wai_r0_version=__version__,
            command=list(command),
            config_hash=config_hash,
            experiment_hash=experiment_hash,
            git_commit=commit,
            git_dirty=dirty,
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    status: GateStatus
    explanation: str
    metric: str | None = None
    observed: float | None = None
    threshold: float | None = None

    def validate(self) -> None:
        if not self.name.strip() or not self.explanation.strip():
            raise ValueError("gate name and explanation cannot be empty")
        for value in (self.observed, self.threshold):
            if value is not None and not math.isfinite(value):
                raise ValueError("gate numeric values must be finite")


@dataclass(slots=True)
class ResearchReport:
    identity: RunIdentity
    evidence_class: EvidenceClass
    resolved_config: dict[str, Any]
    metrics: dict[str, Any]
    gates: list[GateResult]
    decision: Decision = "not_decided"
    limitations: list[str] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    data_manifest: dict[str, Any] | None = None
    tokenizer_manifest: dict[str, Any] | None = None
    failures: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    schema_version: str = REPORT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported report schema: {self.schema_version}")
        if not self.limitations:
            raise ValueError("reports must state at least one limitation")
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise ValueError("reports must contain at least one metric")
        if not self.gates:
            raise ValueError("reports must contain at least one gate")
        for gate in self.gates:
            gate.validate()
        if self.decision == "keep" and any(gate.status == "fail" for gate in self.gates):
            raise ValueError("a report with a failed gate cannot decide 'keep'")
        _reject_non_finite(self.metrics, path="metrics")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResearchReport:
        identity_payload = payload.get("identity")
        gate_payload = payload.get("gates")
        if not isinstance(identity_payload, dict) or not isinstance(gate_payload, list):
            raise ValueError("report identity and gates are required")
        report = cls(
            **{
                **payload,
                "identity": RunIdentity(**identity_payload),
                "gates": [GateResult(**item) for item in gate_payload],
            }
        )
        report.validate()
        return report


def default_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return info


def default_software_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "wai_r0": __version__,
    }


def write_report(path: str | Path, report: ResearchReport) -> Path:
    return atomic_write_json(path, report.to_dict())


def load_report(path: str | Path) -> ResearchReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report root must be a mapping")
    return ResearchReport.from_dict(payload)


def _reject_non_finite(value: Any, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, path=f"{path}[{index}]")


def _git_state(repository: str | Path | None) -> tuple[str | None, bool | None]:
    if repository is None:
        return None, None
    root = Path(repository)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())
