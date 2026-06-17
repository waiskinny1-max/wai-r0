from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re

import yaml

from wai_r0.benchmarks import tiny_train
from wai_r0.training.language_csv import csv_language_probe_report
from wai_r0.config import ReasonerConfig
from wai_r0.report import BenchmarkReport

_ALLOWED_TASKS = {"copy", "reverse", "parity"}
_TINY_MODES = {"tiny_probe", "tiny-train", "tiny_train"}
_CSV_MODES = {"csv_language", "language_csv", "csv-language", "language-csv", "csv"}
_ALLOWED_MODES = _TINY_MODES | _CSV_MODES
_ALLOWED_KEYS = {
    "mode",
    "config",
    "task",
    "examples",
    "batch_size",
    "train_len",
    "eval_lens",
    "output",
    "csv_path",
    "path",
    "text_column",
    "target_column",
    "steps",
    "seq_len",
    "max_rows",
    "lr",
    "eval_rows",
    "eval_interval",
    "baseline_rows",
    "train_fraction",
    "val_fraction",
    "test_fraction",
    "split_seed",
    "checkpoint",
    "checkpoint_path",
    "resume_from",
    "log",
    "log_path",
}


@dataclass(frozen=True)
class MarkdownTrainingPlan:
    """Validated training plan loaded from Markdown.

    The file is declarative. It may configure either a tiny algorithmic probe or
    a CSV language probe, but it cannot execute arbitrary Python, shell commands,
    or unrestricted training behavior.
    """

    source: str
    mode: str = "tiny_probe"
    config: str = "configs/model/nano.yaml"
    output: str | None = None

    # Tiny algorithmic probe fields.
    task: str = "copy"
    examples: int = 32
    batch_size: int = 4
    train_len: int = 8
    eval_lens: tuple[int, ...] = (8, 16)

    # CSV language probe fields.
    csv_path: str | None = None
    text_column: str | None = None
    target_column: str | None = None
    steps: int = 25
    seq_len: int = 64
    max_rows: int | None = None
    lr: float = 3e-4
    eval_rows: int = 8
    eval_interval: int = 5
    baseline_rows: int = 256
    train_fraction: float = 0.90
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    split_seed: int | None = None
    checkpoint_path: str | None = None
    resume_from: str | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eval_lens"] = list(self.eval_lens)
        return data

def _coerce_scalar(value: str) -> Any:
    text = value.strip().strip("'").strip('"')
    if not text:
        return ""
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        loaded = None
    if isinstance(loaded, (str, int, float, bool, list, tuple)) or loaded is None:
        return loaded
    return text

def _extract_frontmatter(markdown: str) -> dict[str, Any] | None:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", markdown, flags=re.DOTALL)
    if not match:
        return None
    loaded = yaml.safe_load(match.group(1)) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Markdown frontmatter must be a mapping.")
    return loaded


def _extract_first_yaml_block(markdown: str) -> dict[str, Any] | None:
    match = re.search(r"```(?:yaml|yml)\s*\n(.*?)\n```", markdown, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    loaded = yaml.safe_load(match.group(1)) or {}
    if not isinstance(loaded, dict):
        raise ValueError("First YAML code block must be a mapping.")
    return loaded


def _extract_key_value_lines(markdown: str) -> dict[str, Any]:
    """Parse simple Markdown list/table-free key-value lines.

    Supported examples:
    - task: copy
    - examples: 32
    task: reverse
    """

    out: dict[str, Any] = {}
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if line.startswith(('-', '*')):
            line = line[1:].strip()
        match = re.match(r"([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1).replace("-", "_")
        if key in _ALLOWED_KEYS:
            out[key] = _coerce_scalar(match.group(2))
    return out


def _as_positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer, not a boolean.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer.") from exc
    if number <= 0:
        raise ValueError(f"{key} must be > 0.")
    return number



def _as_optional_positive_int(value: Any, key: str) -> int | None:
    if value is None or value == "":
        return None
    return _as_positive_int(value, key)


def _as_positive_float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive number, not a boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive number.") from exc
    if number <= 0:
        raise ValueError(f"{key} must be > 0.")
    return number


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _as_eval_lens(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("eval_lens must be an integer, comma string, or list of integers.")
    lens = tuple(_as_positive_int(item, "eval_lens") for item in value)
    if not lens:
        raise ValueError("eval_lens must include at least one length.")
    return lens


def load_markdown_training_plan(path: str | Path) -> MarkdownTrainingPlan:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"training plan not found: {source}")
    if not source.is_file():
        raise ValueError(f"training plan path is not a file: {source}")
    markdown = source.read_text(encoding="utf-8")
    data = _extract_frontmatter(markdown) or _extract_first_yaml_block(markdown) or _extract_key_value_lines(markdown)
    if not data:
        raise ValueError(
            "training plan must contain YAML frontmatter, a YAML code block, or key-value lines "
            "such as 'task: copy' and 'examples: 32'."
        )

    normalized = {str(key).replace("-", "_"): value for key, value in data.items()}
    unknown = sorted(set(normalized) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unsupported training plan keys: {', '.join(unknown)}")

    mode = str(normalized.get("mode", "tiny_probe")).strip()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            "unsupported training mode. Supported modes: "
            f"{', '.join(sorted(_ALLOWED_MODES))}. Full language pretraining is not implemented."
        )

    config = str(normalized.get("config", "configs/model/nano.yaml")).strip()
    output = _optional_str(normalized.get("output"))

    if mode in _CSV_MODES:
        csv_path = _optional_str(normalized.get("csv_path") or normalized.get("path"))
        if csv_path is None:
            raise ValueError("CSV language mode requires csv_path, e.g. csv_path: training/basic_lang.csv")
        return MarkdownTrainingPlan(
            source=str(source),
            mode="csv_language",
            config=config,
            output=output,
            batch_size=_as_positive_int(normalized.get("batch_size", 4), "batch_size"),
            csv_path=csv_path,
            text_column=_optional_str(normalized.get("text_column")),
            target_column=_optional_str(normalized.get("target_column")),
            steps=_as_positive_int(normalized.get("steps", 25), "steps"),
            seq_len=_as_positive_int(normalized.get("seq_len", 64), "seq_len"),
            max_rows=_as_optional_positive_int(normalized.get("max_rows"), "max_rows"),
            lr=_as_positive_float(normalized.get("lr", 3e-4), "lr"),
            eval_rows=_as_positive_int(normalized.get("eval_rows", 8), "eval_rows"),
            eval_interval=_as_positive_int(normalized.get("eval_interval", 5), "eval_interval"),
            baseline_rows=_as_positive_int(normalized.get("baseline_rows", 256), "baseline_rows"),
            train_fraction=_as_positive_float(normalized.get("train_fraction", 0.90), "train_fraction"),
            val_fraction=_as_positive_float(normalized.get("val_fraction", 0.05), "val_fraction"),
            test_fraction=_as_positive_float(normalized.get("test_fraction", 0.05), "test_fraction"),
            split_seed=_as_optional_positive_int(normalized.get("split_seed"), "split_seed"),
            checkpoint_path=_optional_str(normalized.get("checkpoint_path") or normalized.get("checkpoint")),
            resume_from=_optional_str(normalized.get("resume_from")),
            log_path=_optional_str(normalized.get("log_path") or normalized.get("log")),
        )

    task = str(normalized.get("task", "copy")).strip()
    if task not in _ALLOWED_TASKS:
        raise ValueError(f"unsupported task: {task}. Supported tasks: {', '.join(sorted(_ALLOWED_TASKS))}.")

    return MarkdownTrainingPlan(
        source=str(source),
        mode="tiny_probe",
        config=config,
        output=output,
        task=task,
        examples=_as_positive_int(normalized.get("examples", 32), "examples"),
        batch_size=_as_positive_int(normalized.get("batch_size", 4), "batch_size"),
        train_len=_as_positive_int(normalized.get("train_len", 8), "train_len"),
        eval_lens=_as_eval_lens(normalized.get("eval_lens", (8, 16))),
    )

def run_markdown_training_plan(path: str | Path) -> tuple[BenchmarkReport, MarkdownTrainingPlan]:
    plan = load_markdown_training_plan(path)

    if plan.mode == "csv_language":
        if plan.csv_path is None:
            raise ValueError("csv_language plan is missing csv_path")
        report = csv_language_probe_report(
            ReasonerConfig.from_yaml(plan.config),
            csv_path=plan.csv_path,
            text_column=plan.text_column,
            target_column=plan.target_column,
            steps=plan.steps,
            batch_size=plan.batch_size,
            seq_len=plan.seq_len,
            max_rows=plan.max_rows,
            lr=plan.lr,
            eval_rows=plan.eval_rows,
            checkpoint_path=plan.checkpoint_path,
            log_path=plan.log_path,
            resume_from=plan.resume_from,
            eval_interval=plan.eval_interval,
            train_fraction=plan.train_fraction,
            val_fraction=plan.val_fraction,
            test_fraction=plan.test_fraction,
            split_seed=plan.split_seed,
            baseline_rows=plan.baseline_rows,
        )
        report.name = "train_md_csv"
        report.benchmark_config = {
            **report.benchmark_config,
            "training_plan": plan.to_dict(),
        }
        report.raw_metrics = {
            **report.raw_metrics,
            "training_plan_source": plan.source,
            "training_plan_mode": plan.mode,
        }
        return report, plan

    report = tiny_train(
        ReasonerConfig.from_yaml(plan.config),
        task=plan.task,
        examples=plan.examples,
        batch_size=plan.batch_size,
        train_len=plan.train_len,
        eval_lens=plan.eval_lens,
    )
    report.name = "train_md"
    report.benchmark_config = {
        **report.benchmark_config,
        "training_plan": plan.to_dict(),
    }
    report.raw_metrics = {
        **report.raw_metrics,
        "training_plan_source": plan.source,
        "training_plan_mode": plan.mode,
    }
    report.summary = (
        "Markdown-configured tiny-training architecture probe completed. "
        "This is not language pretraining and not evidence that random weights reason."
    )
    report.limitations = [
        "The -train Markdown entrypoint supports tiny_probe and csv_language probe modes only.",
        *report.limitations,
    ]
    return report, plan
