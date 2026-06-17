from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator
import csv
import math

import torch
import torch.nn.functional as F

from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore, set_seed
from wai_r0.report import BenchmarkReport

BYTE_VOCAB_SIZE = 259
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3

_PROMPT_COLUMNS = ("prompt", "instruction", "input", "question", "source")
_TARGET_COLUMNS = ("completion", "response", "output", "answer", "target")
_TEXT_COLUMNS = ("text", "content", "sentence", "sample")


@dataclass(frozen=True)
class CSVInspection:
    path: str
    exists: bool
    header: list[str]
    detected_text_column: str | None
    detected_target_column: str | None
    sampled_rows: int
    nonempty_rows: int
    min_chars: int
    max_chars: int
    mean_chars: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CSVLanguageProbeResult:
    csv_path: str
    text_column: str
    target_column: str | None
    steps: int
    batch_size: int
    seq_len: int
    max_rows: int | None
    rows_consumed: int
    initial_loss: float
    final_loss: float
    loss_delta: float
    eval_loss: float
    eval_token_accuracy: float
    byte_level_perplexity: float
    checkpoint_path: str | None
    inspection: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ByteTokenizer:
    """Minimal byte-level tokenizer for CSV smoke training.

    This is deliberately local and dependency-free. It is not BPE, SentencePiece,
    or a production tokenizer; it only gives WAI-R0 a stable way to run a tiny
    causal-language probe against arbitrary UTF-8 CSV text.
    """

    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    vocab_size = BYTE_VOCAB_SIZE

    def encode(self, text: str, max_tokens: int) -> list[int]:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2 to include BOS/EOS tokens.")
        payload = text.encode("utf-8", errors="replace")[: max_tokens - 2]
        return [self.bos_id, *[int(byte) + BYTE_OFFSET for byte in payload], self.eos_id]

    def pad(self, ids: list[int], length: int) -> list[int]:
        if len(ids) >= length:
            return ids[:length]
        return [*ids, *([self.pad_id] * (length - len(ids)))]


def language_ready_config(cfg: ReasonerConfig) -> ReasonerConfig:
    """Return a config with enough vocabulary for byte-level language probes."""

    if cfg.vocab_size >= BYTE_VOCAB_SIZE:
        return cfg
    return ReasonerConfig.from_dict({**cfg.to_dict(), "vocab_size": BYTE_VOCAB_SIZE})


def _sniff_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc
    normalized = [str(column).strip() for column in header]
    if not normalized or all(not column for column in normalized):
        raise ValueError("CSV header row is empty. Add a header such as text or prompt,completion.")
    return normalized


def _first_present(header: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_real = {column.lower(): column for column in header}
    for candidate in candidates:
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    return None


def detect_language_columns(
    header: list[str],
    text_column: str | None = None,
    target_column: str | None = None,
) -> tuple[str, str | None]:
    if text_column:
        if text_column not in header:
            raise ValueError(f"text column '{text_column}' not found. Available columns: {', '.join(header)}")
        detected_text = text_column
    else:
        detected_text = (
            _first_present(header, _TEXT_COLUMNS)
            or _first_present(header, _PROMPT_COLUMNS)
            or header[0]
        )

    if target_column:
        if target_column not in header:
            raise ValueError(f"target column '{target_column}' not found. Available columns: {', '.join(header)}")
        detected_target = target_column
    else:
        detected_target = _first_present([column for column in header if column != detected_text], _TARGET_COLUMNS)
    return detected_text, detected_target


def _row_text(row: dict[str, str], text_column: str, target_column: str | None) -> str:
    text = str(row.get(text_column, "") or "").strip()
    if target_column is None:
        return text
    target = str(row.get(target_column, "") or "").strip()
    if text and target:
        return f"{text}\n{target}"
    return text or target


def iter_language_texts(
    path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
) -> Iterator[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    header = _sniff_header(csv_path)
    text_col, target_col = detect_language_columns(header, text_column, target_column)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for row in reader:
            if max_rows is not None and count >= max_rows:
                break
            text = _row_text(row, text_col, target_col)
            if not text:
                continue
            count += 1
            yield text


def inspect_language_csv(
    path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    sample_rows: int = 1000,
) -> CSVInspection:
    csv_path = Path(path)
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    if not csv_path.exists():
        return CSVInspection(str(csv_path), False, [], None, None, 0, 0, 0, 0, 0.0, ["file does not exist"])

    header = _sniff_header(csv_path)
    text_col, target_col = detect_language_columns(header, text_column, target_column)
    lengths: list[int] = []
    for text in islice(iter_language_texts(csv_path, text_col, target_col, max_rows=sample_rows), sample_rows):
        if text:
            lengths.append(len(text))

    warnings: list[str] = []
    if not lengths:
        warnings.append("no nonempty training rows found in sampled CSV rows")
    if text_column is None and text_col == header[0] and text_col.lower() not in {*_TEXT_COLUMNS, *_PROMPT_COLUMNS}:
        warnings.append("text column was inferred from the first CSV column; pass --text-column to make this explicit")
    if target_col is None and any(column.lower() in _TARGET_COLUMNS for column in header):
        warnings.append("a target-like column exists but was not paired; pass --target-column if this is supervised prompt/completion data")

    sampled = len(lengths)
    return CSVInspection(
        path=str(csv_path),
        exists=True,
        header=header,
        detected_text_column=text_col,
        detected_target_column=target_col,
        sampled_rows=sampled,
        nonempty_rows=sampled,
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
        mean_chars=(sum(lengths) / sampled) if sampled else 0.0,
        warnings=warnings,
    )


def _batch_from_texts(texts: list[str], tokenizer: ByteTokenizer, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4 for language probes.")
    encoded = [tokenizer.pad(tokenizer.encode(text, seq_len + 1), seq_len + 1) for text in texts]
    tokens = torch.tensor(encoded, dtype=torch.long, device=device)
    x = tokens[:, :-1].contiguous()
    y = tokens[:, 1:].contiguous()
    y = y.masked_fill(y == tokenizer.pad_id, -100)
    return x, y


def _next_text_batch(iterator: Iterator[str], batch_size: int) -> list[str]:
    batch: list[str] = []
    for _ in range(batch_size):
        try:
            batch.append(next(iterator))
        except StopIteration:
            break
    return batch


def _token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    valid = target.ne(-100)
    if not bool(valid.any().item()):
        return 0.0
    pred = logits.argmax(dim=-1)
    return float((pred.eq(target) & valid).float().sum().item() / valid.float().sum().item())


def _evaluate(core: ReasonerCore, rows: list[str], tokenizer: ByteTokenizer, seq_len: int) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    core.eval()
    with torch.no_grad():
        x, y = _batch_from_texts(rows, tokenizer, seq_len, core.device_obj)
        logits = core(x, mode="think" if core.cfg.recurrent_depth > 1 else "fast")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        return float(loss.item()), _token_accuracy(logits, y)


def _make_training_iterator(
    path: Path,
    text_column: str,
    target_column: str | None,
    max_rows: int | None,
) -> Iterator[str]:
    while True:
        yielded = False
        for text in iter_language_texts(path, text_column, target_column, max_rows=max_rows):
            yielded = True
            yield text
        if not yielded:
            raise ValueError("CSV did not yield any nonempty training rows.")


def run_csv_language_probe(
    cfg: ReasonerConfig,
    csv_path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    steps: int = 25,
    batch_size: int = 4,
    seq_len: int = 64,
    max_rows: int | None = None,
    lr: float = 3e-4,
    eval_rows: int = 8,
    checkpoint_path: str | Path | None = None,
) -> CSVLanguageProbeResult:
    """Run a small byte-level CSV language probe.

    This streams CSV rows and performs causal next-byte prediction. It is a
    deliberately tiny probe for local architecture iteration, not full language
    pretraining and not evidence of semantic reasoning.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if eval_rows <= 0:
        raise ValueError("eval_rows must be positive.")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")

    path = Path(csv_path)
    inspection = inspect_language_csv(path, text_column, target_column, sample_rows=min(max_rows or 1000, 1000))
    if not inspection.exists:
        raise FileNotFoundError(f"CSV not found: {path}")
    if inspection.nonempty_rows == 0:
        raise ValueError("CSV has no nonempty language rows in the inspected sample.")
    text_col = inspection.detected_text_column
    target_col = inspection.detected_target_column
    if text_col is None:
        raise ValueError("could not detect text column")

    language_cfg = language_ready_config(cfg)
    safe_seq_len = min(seq_len, language_cfg.max_seq_len)
    set_seed(language_cfg.seed)
    core = ReasonerCore(language_cfg)
    tokenizer = ByteTokenizer()
    opt = torch.optim.AdamW(core.parameters(), lr=lr)

    eval_sample = list(islice(iter_language_texts(path, text_col, target_col, max_rows=max_rows), eval_rows))
    initial_eval_loss, _ = _evaluate(core, eval_sample, tokenizer, safe_seq_len)

    iterator = _make_training_iterator(path, text_col, target_col, max_rows)
    final_loss = 0.0
    rows_consumed = 0
    core.train()
    for _ in range(steps):
        batch = _next_text_batch(iterator, batch_size)
        if len(batch) < batch_size:
            iterator = _make_training_iterator(path, text_col, target_col, max_rows)
            batch.extend(_next_text_batch(iterator, batch_size - len(batch)))
        rows_consumed += len(batch)
        x, y = _batch_from_texts(batch, tokenizer, safe_seq_len, core.device_obj)
        opt.zero_grad(set_to_none=True)
        logits = core(x, mode="think" if language_cfg.recurrent_depth > 1 else "fast")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.item())

    eval_loss, eval_acc = _evaluate(core, eval_sample, tokenizer, safe_seq_len)
    checkpoint_out: str | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": core.state_dict(),
                "model_config": language_cfg.to_dict(),
                "tokenizer": {
                    "type": "byte-level",
                    "pad_id": PAD_ID,
                    "bos_id": BOS_ID,
                    "eos_id": EOS_ID,
                    "byte_offset": BYTE_OFFSET,
                    "vocab_size": BYTE_VOCAB_SIZE,
                },
                "training": {
                    "csv_path": str(path),
                    "text_column": text_col,
                    "target_column": target_col,
                    "steps": steps,
                    "batch_size": batch_size,
                    "seq_len": safe_seq_len,
                    "max_rows": max_rows,
                    "lr": lr,
                },
            },
            checkpoint,
        )
        checkpoint_out = str(checkpoint)

    return CSVLanguageProbeResult(
        csv_path=str(path),
        text_column=text_col,
        target_column=target_col,
        steps=steps,
        batch_size=batch_size,
        seq_len=safe_seq_len,
        max_rows=max_rows,
        rows_consumed=rows_consumed,
        initial_loss=float(initial_eval_loss),
        final_loss=float(final_loss),
        loss_delta=float(initial_eval_loss - eval_loss),
        eval_loss=float(eval_loss),
        eval_token_accuracy=float(eval_acc),
        byte_level_perplexity=float(math.exp(min(eval_loss, 20.0))) if eval_loss > 0 else 0.0,
        checkpoint_path=checkpoint_out,
        inspection=inspection.to_dict(),
    )


def csv_language_probe_report(
    cfg: ReasonerConfig,
    csv_path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    steps: int = 25,
    batch_size: int = 4,
    seq_len: int = 64,
    max_rows: int | None = None,
    lr: float = 3e-4,
    eval_rows: int = 8,
    checkpoint_path: str | Path | None = None,
) -> BenchmarkReport:
    """Return a standard WAI-R0 report for a CSV language probe.

    The report label is intentionally conservative. The probe measures whether a
    small architecture can reduce byte-level next-token loss on user-provided CSV
    language data. It does not claim semantic understanding or reasoning.
    """

    result = run_csv_language_probe(
        cfg=cfg,
        csv_path=csv_path,
        text_column=text_column,
        target_column=target_column,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        max_rows=max_rows,
        lr=lr,
        eval_rows=eval_rows,
        checkpoint_path=checkpoint_path,
    )
    loss_improved = result.loss_delta > 0
    return BenchmarkReport(
        name="csv_language_probe",
        result_type="tiny-training CSV language probe",
        seed=cfg.seed,
        device=cfg.device,
        dtype=cfg.dtype,
        model_config=language_ready_config(cfg).to_dict(),
        benchmark_config={
            "csv_path": str(csv_path),
            "text_column": text_column,
            "target_column": target_column,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "max_rows": max_rows,
            "lr": lr,
            "eval_rows": eval_rows,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        },
        raw_metrics=result.to_dict(),
        summary=(
            "CSV language probe completed. The run streams rows from CSV and trains a byte-level "
            "causal next-token objective. Loss improvement is a training-readiness signal, not proof "
            "of language understanding or reasoning."
        ),
        limitations=[
            "This uses a dependency-free byte tokenizer, not BPE/SentencePiece.",
            "The CSV is streamed for training, but evaluation uses a small held sample.",
            "A lower loss only means the tiny model adapted to byte statistics under this budget.",
            "This is not full pretraining, not instruction tuning, and not evidence of semantic reasoning.",
        ],
        recommendation=(
            "TINY-TRAIN ONLY — CSV probe improved loss; run larger controlled baselines next."
            if loss_improved
            else "DO NOT SCALE YET — CSV probe did not improve loss under this tiny budget."
        ),
    )
