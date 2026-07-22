from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY_SOURCE_PATHS = {
    "src/wai_r0/benchmarks.py",
    "src/wai_r0/cli.py",
    "src/wai_r0/gui.py",
    "src/wai_r0/report.py",
    "src/wai_r0/symbolic.py",
    "src/wai_r0/eval/holdout.py",
    "src/wai_r0/eval/leakage_guard.py",
    "src/wai_r0/eval/prior_diagnostics.py",
    "src/wai_r0/eval/scorecard.py",
    "src/wai_r0/eval/suite.py",
    "src/wai_r0/training/language_csv.py",
    "src/wai_r0/training/markdown_plan.py",
    "src/wai_r0/training/probes.py",
}


def _native_source_paths() -> list[Path]:
    return [
        path
        for path in sorted((ROOT / "src/wai_r0").rglob("*.py"))
        if str(path.relative_to(ROOT)).replace("\\", "/") not in LEGACY_SOURCE_PATHS
    ]


def _quality_paths() -> list[Path]:
    return [
        *_native_source_paths(),
        ROOT / "main.py",
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tests").glob("test_v0[56]_*.py")),
        ROOT / "tests/conftest.py",
    ]


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    source = _native_source_paths()
    all_paths = _quality_paths()
    missing = [str(path.relative_to(ROOT)) for path in all_paths if not path.is_file()]
    if missing:
        raise SystemExit("missing quality-gate paths: " + ", ".join(missing))
    source_names = [str(path.relative_to(ROOT)) for path in source]
    all_names = [str(path.relative_to(ROOT)) for path in all_paths]
    _run(["ruff", "format", "--check", *all_names])
    _run(["ruff", "check", *all_names])
    _run(["mypy", "--no-incremental", "--follow-imports=skip", *source_names])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
