from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V05_SOURCE_FILES = [
    ROOT / "src/wai_r0/__init__.py",
    ROOT / "src/wai_r0/config.py",
    ROOT / "src/wai_r0/model.py",
    ROOT / "src/wai_r0/profiler.py",
    ROOT / "src/wai_r0/v05_cli.py",
    ROOT / "src/wai_r0/version.py",
]
V05_SOURCE_DIRS = [
    ROOT / "src/wai_r0/app",
    ROOT / "src/wai_r0/core",
    ROOT / "src/wai_r0/data",
    ROOT / "src/wai_r0/modeling",
    ROOT / "src/wai_r0/experiments",
    ROOT / "src/wai_r0/reporting",
]
V05_EVAL_FILES = [
    ROOT / "src/wai_r0/eval/__init__.py",
    ROOT / "src/wai_r0/eval/algorithmic.py",
    ROOT / "src/wai_r0/eval/gates.py",
    ROOT / "src/wai_r0/eval/metrics.py",
]
V05_TRAINING_FILES = [
    ROOT / "src/wai_r0/training/__init__.py",
    ROOT / "src/wai_r0/training/checkpoint.py",
    ROOT / "src/wai_r0/training/engine.py",
    ROOT / "src/wai_r0/training/losses.py",
    ROOT / "src/wai_r0/training/optimizer.py",
    ROOT / "src/wai_r0/training/schedules.py",
]


def _source_paths() -> list[Path]:
    paths = [*V05_SOURCE_FILES, *V05_EVAL_FILES, *V05_TRAINING_FILES]
    for directory in V05_SOURCE_DIRS:
        paths.extend(sorted(directory.glob("*.py")))
    return sorted(set(paths))


def _non_source_quality_paths() -> list[Path]:
    return [
        ROOT / "scripts/check_v05_quality.py",
        ROOT / "tests/conftest.py",
        *sorted((ROOT / "tests").glob("test_v05_*.py")),
    ]


def _validate(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing quality-gate paths: " + ", ".join(missing))


def _run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"+ {printable}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    source_paths = _source_paths()
    all_paths = [*source_paths, *_non_source_quality_paths()]
    _validate(all_paths)
    relative_source = [str(path.relative_to(ROOT)) for path in source_paths]
    relative_all = [str(path.relative_to(ROOT)) for path in all_paths]
    _run(["ruff", "format", "--check", *relative_all])
    _run(["ruff", "check", *relative_all])
    _run(
        [
            "mypy",
            "--no-incremental",
            "--follow-imports=skip",
            *relative_source,
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
