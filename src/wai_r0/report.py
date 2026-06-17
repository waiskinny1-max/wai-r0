from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import platform
import subprocess

import torch


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def recommend(score: float) -> str:
    if score < 0.35:
        return "DO NOT TRAIN — architecture has no useful signal."
    if score < 0.60:
        return "TINY-TRAIN ONLY — promising but unproven."
    if score < 0.80:
        return "SCALE TO SMALL — worth 150M–350M experiment."
    return "SCALE CAREFULLY — worth serious pretraining investigation."


@dataclass
class BenchmarkReport:
    name: str
    result_type: str
    seed: int
    device: str
    dtype: str
    model_config: dict[str, Any] = field(default_factory=dict)
    benchmark_config: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    limitations: list[str] = field(default_factory=list)
    recommendation: str = "TINY-TRAIN ONLY — promising but unproven."
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = field(default_factory=git_commit)
    hardware: dict[str, Any] = field(default_factory=hardware)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        limitations = "\n".join(f"- {item}" for item in self.limitations) or "- None recorded."
        metrics = json.dumps(self.raw_metrics, indent=2, sort_keys=True)
        return f"""# {self.name}

## Metadata

- Date: {self.date}
- Git commit: {self.git_commit or 'unavailable'}
- Device: {self.device}
- Dtype: {self.dtype}
- Seed: {self.seed}
- Result type: {self.result_type}

## Summary

{self.summary}

## Raw metrics

```json
{metrics}
```

## Limitations

{limitations}

## Recommendation

{self.recommendation}
"""

    def write(self, outdir: str | Path = "reports", base: str = "latest") -> tuple[Path, Path]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{base}.json"
        md_path = out / f"{base}.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path


def markdown_from_json(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = BenchmarkReport.__dataclass_fields__
    return BenchmarkReport(**{key: value for key, value in data.items() if key in fields}).to_markdown()
