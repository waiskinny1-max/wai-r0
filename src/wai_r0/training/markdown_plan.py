from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re

import yaml

from wai_r0.benchmarks import tiny_train
from wai_r0.config import ReasonerConfig
from wai_r0.report import BenchmarkReport

_ALLOWED_TASKS = {"copy", "reverse", "parity"}
_ALLOWED_MODES = {"tiny_probe", "tiny-train", "tiny_train"}
_ALLOWED_KEYS = {
    "mode",
    "config",
    "task",
    "examples",
    "batch_size",
    "train_len",
    "eval_lens",
    "output",
}


@dataclass(frozen=True)
class MarkdownTrainingPlan:
    """Validated training plan loaded from Markdown.

    This is intentionally narrow. A Markdown file may configure a tiny-training
    architecture probe, but it cannot execute arbitrary Python, shell commands,
    or unrestricted training behavior.
    """

    source: str
    mode: str = "tiny_probe"
    config: str = "configs/model/nano.yaml"
    task: str = "copy"
    examples: int = 32
    batch_size: int = 4
    train_len: int = 8
    eval_lens: tuple[int, ...] = (8, 16)
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eval_lens"] = list(self.eval_lens)
        return data


def _coerce_scalar(value: str) -> Any:
    text = value.strip().strip("'").strip('"')
    if not text:
        return ""
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if "," in text and not (text.startswith("http://") or text.startswith("https://")):
        return [part.strip() for part in text.split(",") if part.strip()]
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
        raise ValueError("only mode='tiny_probe' is supported in this version; full pretraining is not implemented.")

    task = str(normalized.get("task", "copy")).strip()
    if task not in _ALLOWED_TASKS:
        raise ValueError(f"unsupported task: {task}. Supported tasks: {', '.join(sorted(_ALLOWED_TASKS))}.")

    config = str(normalized.get("config", "configs/model/nano.yaml")).strip()
    output = normalized.get("output")
    return MarkdownTrainingPlan(
        source=str(source),
        mode="tiny_probe",
        config=config,
        task=task,
        examples=_as_positive_int(normalized.get("examples", 32), "examples"),
        batch_size=_as_positive_int(normalized.get("batch_size", 4), "batch_size"),
        train_len=_as_positive_int(normalized.get("train_len", 8), "train_len"),
        eval_lens=_as_eval_lens(normalized.get("eval_lens", (8, 16))),
        output=str(output).strip() if output is not None and str(output).strip() else None,
    )


def run_markdown_training_plan(path: str | Path) -> tuple[BenchmarkReport, MarkdownTrainingPlan]:
    plan = load_markdown_training_plan(path)
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
        "The -train Markdown entrypoint currently supports tiny_probe mode only.",
        *report.limitations,
    ]
    return report, plan
