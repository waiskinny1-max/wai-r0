from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wai_r0.benchmarks import architecture_priors, memory, symbolic, tiny_train, zero_neural
from wai_r0.config import ReasonerConfig
from wai_r0.report import BenchmarkReport


@dataclass(frozen=True)
class SuiteRunResult:
    output_dir: Path
    reports: list[BenchmarkReport]
    written: list[tuple[Path, Path]]

    def summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "reports": [report.name for report in self.reports],
            "written": [[str(json_path), str(md_path)] for json_path, md_path in self.written],
        }


def _load_suite(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"output_dir": "reports/suite_v03", "steps": [{"name": "zero"}, {"name": "prior"}, {"name": "memory"}]}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("suite config must be a YAML mapping")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("suite config must contain a non-empty steps list")
    return raw


def _step_name(step: dict[str, Any]) -> str:
    name = step.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("each suite step requires a non-empty name")
    return name


def run_suite(cfg: ReasonerConfig, suite_path: str | Path | None = None) -> SuiteRunResult:
    suite = _load_suite(suite_path)
    output_dir = Path(str(suite.get("output_dir", "reports/suite_v03")))
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BenchmarkReport] = []
    written: list[tuple[Path, Path]] = []

    for raw_step in suite["steps"]:
        if not isinstance(raw_step, dict):
            raise ValueError("suite steps must be mappings")
        name = _step_name(raw_step)
        if name == "zero":
            report = zero_neural(
                cfg,
                batch_size=int(raw_step.get("batch_size", 1)),
                seq_len=int(raw_step.get("seq_len", 8)),
            )
        elif name == "prior":
            report = architecture_priors(
                cfg,
                batch_size=int(raw_step.get("batch_size", 2)),
                seq_len=int(raw_step.get("seq_len", 16)),
                recurrent_depths=tuple(int(v) for v in raw_step.get("recurrent_depths", [1, 2, 4])),
            )
        elif name == "memory":
            seq_lens = [int(v) for v in raw_step.get("seq_lens", [32, 64])]
            report = memory(
                cfg,
                baseline=str(raw_step.get("baseline", "mha")),
                candidate=str(raw_step.get("candidate", "mla_lite")),
                seq_lens=seq_lens,
                batch_size=int(raw_step.get("batch_size", 1)),
            )
        elif name == "symbolic":
            report = symbolic(
                raw_step.get("tasks", "examples/tasks"),
                budget_s=float(raw_step.get("budget_s", 3.0)),
                max_depth=int(raw_step.get("max_depth", 2)),
                leakage_manifest=raw_step.get("leakage_manifest"),
                split=str(raw_step.get("split", "dev")),
                register_leakage=bool(raw_step.get("register_leakage", False)),
            )
        elif name == "tiny":
            report = tiny_train(
                cfg,
                task=str(raw_step.get("task", "copy")),
                examples=int(raw_step.get("examples", 8)),
                batch_size=int(raw_step.get("batch_size", 4)),
                train_len=int(raw_step.get("train_len", 8)),
                eval_lens=tuple(int(v) for v in raw_step.get("eval_lens", [8, 16])),
            )
        else:
            raise ValueError(f"unsupported suite step: {name}")
        reports.append(report)
        written.append(report.write(output_dir, name))
    return SuiteRunResult(output_dir=output_dir, reports=reports, written=written)
