from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wai_r0.core.reproducibility import atomic_write_json
from wai_r0.data.csv_reader import iter_conversation_rows
from wai_r0.tokenization.bpe import BPETrainingSummary, train_deterministic_bpe
from wai_r0.tokenization.normalization import NormalizationMode


@dataclass(frozen=True, slots=True)
class TokenizerTrainingConfig:
    vocab_size: int = 4096
    min_frequency: int = 2
    normalization: NormalizationMode = "none"
    max_rows: int | None = None
    max_training_bytes: int = 16_000_000

    def validate(self) -> None:
        if self.vocab_size < 261:
            raise ValueError("vocab_size must be at least 261")
        if self.min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError("max_rows must be positive when set")
        if self.max_training_bytes < 1:
            raise ValueError("max_training_bytes must be positive")


@dataclass(frozen=True, slots=True)
class TokenizerTrainingResult:
    tokenizer_path: str
    summary_path: str
    manifest_hash: str
    summary: BPETrainingSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_path": self.tokenizer_path,
            "summary_path": self.summary_path,
            "manifest_hash": self.manifest_hash,
            "summary": self.summary.to_dict(),
        }


def train_bpe_from_conversation_csv(
    csv_path: str | Path,
    *,
    output: str | Path,
    config: TokenizerTrainingConfig | None = None,
) -> TokenizerTrainingResult:
    active = config or TokenizerTrainingConfig()
    active.validate()

    def corpus() -> Any:
        for row in iter_conversation_rows(csv_path, max_rows=active.max_rows):
            if row.system:
                yield row.system
            yield row.user
            yield row.assistant

    tokenizer, summary = train_deterministic_bpe(
        corpus(),
        vocab_size=active.vocab_size,
        min_frequency=active.min_frequency,
        normalization=active.normalization,
        max_training_bytes=active.max_training_bytes,
    )
    destination = Path(output)
    tokenizer_path = tokenizer.save(destination)
    summary_path = atomic_write_json(
        destination.with_suffix(destination.suffix + ".training.json"),
        {"config": asdict(active), "summary": summary.to_dict()},
    )
    manifest = tokenizer.manifest()
    return TokenizerTrainingResult(
        tokenizer_path=str(tokenizer_path),
        summary_path=str(summary_path),
        manifest_hash=str(manifest["manifest_hash"]),
        summary=summary,
    )


__all__ = [
    "TokenizerTrainingConfig",
    "TokenizerTrainingResult",
    "train_bpe_from_conversation_csv",
]
