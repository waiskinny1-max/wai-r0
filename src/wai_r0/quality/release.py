from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from wai_r0.hardware.capabilities import runtime_capabilities
from wai_r0.version import __version__


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    name: str
    status: str
    detail: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseDoctorReport:
    version: str
    repository: str
    checks: list[ReleaseCheck]
    runtime: dict[str, Any]

    @property
    def ready(self) -> bool:
        return all(check.status == "pass" for check in self.checks if check.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "repository": self.repository,
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
            "runtime": self.runtime,
        }


def _git_output(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _check_file(repository: Path, relative: str, *, required: bool = True) -> ReleaseCheck:
    path = repository / relative
    return ReleaseCheck(
        name=f"file:{relative}",
        status="pass" if path.is_file() else "fail",
        detail=str(path),
        required=required,
    )


def inspect_release(repository: str | Path = ".") -> ReleaseDoctorReport:
    root = Path(repository).resolve()
    checks = [
        _check_file(root, "pyproject.toml"),
        _check_file(root, "README.md"),
        _check_file(root, ".github/workflows/ci.yml"),
        _check_file(root, "src/wai_r0/version.py"),
        _check_file(root, "src/wai_r0/app/cli.py"),
        _check_file(root, "docs/SCIENTIFIC_LIMITS.md"),
    ]
    version_file = root / "src/wai_r0/version.py"
    version_consistent = version_file.is_file() and __version__ in version_file.read_text(
        encoding="utf-8"
    )
    checks.append(
        ReleaseCheck(
            "canonical_version",
            "pass" if version_consistent else "fail",
            f"runtime={__version__}",
        )
    )
    workflow = root / ".github/workflows/ci.yml"
    workflow_valid = False
    if workflow.is_file():
        try:
            workflow_payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
            workflow_valid = isinstance(workflow_payload, dict) and bool(
                workflow_payload.get("jobs")
            )
        except (OSError, UnicodeError, yaml.YAMLError):
            workflow_valid = False
    checks.append(
        ReleaseCheck(
            "ci_workflow_parse",
            "pass" if workflow_valid else "fail",
            "CI workflow contains at least one job" if workflow_valid else "invalid CI workflow",
        )
    )
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    checks.append(
        ReleaseCheck(
            "git_repository",
            "pass" if commit else "warn",
            commit or "git metadata unavailable",
            required=False,
        )
    )
    checks.append(
        ReleaseCheck(
            "clean_worktree",
            "pass" if status == "" else ("warn" if status is not None else "unknown"),
            "clean" if status == "" else (status or "git metadata unavailable"),
            required=False,
        )
    )
    tracked_artifacts = _git_output(
        root,
        "ls-files",
        "*.pt",
        "*.pth",
        "*.bin",
        "reports/*",
        "dist/*",
        "build/*",
    )
    checks.append(
        ReleaseCheck(
            "generated_artifacts_untracked",
            "pass" if not tracked_artifacts else "fail",
            tracked_artifacts or "no generated training/build artifacts are tracked",
        )
    )
    runtime = runtime_capabilities()
    runtime["git_commit"] = commit
    runtime["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    return ReleaseDoctorReport(
        version=__version__,
        repository=str(root),
        checks=checks,
        runtime=runtime,
    )


def write_release_report(path: str | Path, report: ReleaseDoctorReport) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "ReleaseCheck",
    "ReleaseDoctorReport",
    "inspect_release",
    "write_release_report",
]
