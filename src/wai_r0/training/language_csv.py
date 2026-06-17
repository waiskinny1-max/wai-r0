from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal
import csv
import hashlib
import json
import math
import time

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
SplitName = Literal["train", "val", "test", "all"]


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
class CSVSplitSpec:
    train: float = 0.90
    val: float = 0.05
    test: float = 0.05
    seed: int = 1337

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split fractions must be non-negative")
        total = sum(values)
        if total <= 0:
            raise ValueError("at least one split fraction must be positive")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CSVCorpusAudit:
    path: str
    header: list[str]
    text_column: str
    target_column: str | None
    max_rows: int | None
    rows_seen: int
    nonempty_rows: int
    duplicate_rows: int
    duplicate_rate: float
    split_counts: dict[str, int]
    min_chars: int
    max_chars: int
    mean_chars: float
    total_chars: int
    total_utf8_bytes: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ByteUnigramBaseline:
    train_rows: int
    eval_rows: int
    eval_tokens: int
    loss: float
    perplexity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CSVTrainingStep:
    step: int
    train_loss: float
    eval_loss: float | None = None
    eval_token_accuracy: float | None = None
    seconds_elapsed: float | None = None

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
    # v0.4-real fields. Kept optional/defaulted so older callers remain stable.
    audit: dict[str, Any] = field(default_factory=dict)
    split_spec: dict[str, Any] = field(default_factory=dict)
    uniform_baseline_loss: float | None = None
    unigram_baseline: dict[str, Any] | None = None
    best_eval_loss: float | None = None
    best_checkpoint_path: str | None = None
    log_path: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    model_parameters: int = 0
    tokens_seen: int = 0
    resumed_from: str | None = None
    recommendation: str = "TINY-TRAIN ONLY — language-probe result must be interpreted conservatively."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ByteTokenizer:
    """Dependency-free byte tokenizer for WAI-R0 language probes.

    This deliberately avoids BPE/SentencePiece so the training path remains
    runnable on a bare Python/PyTorch install. It is correct for diagnostics,
    not for serious language-model training quality.
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

    def decode(self, ids: Iterable[int]) -> str:
        payload = bytearray()
        for token in ids:
            value = int(token)
            if value in {self.pad_id, self.bos_id, self.eos_id}:
                continue
            if value >= BYTE_OFFSET:
                payload.append(max(0, min(255, value - BYTE_OFFSET)))
        return payload.decode("utf-8", errors="replace")

    def pad(self, ids: list[int], length: int) -> list[int]:
        if len(ids) >= length:
            return ids[:length]
        return [*ids, *([self.pad_id] * (length - len(ids)))]


def language_ready_config(cfg: ReasonerConfig) -> ReasonerConfig:
    """Return a config with enough vocabulary for byte-level probes."""

    if cfg.vocab_size >= BYTE_VOCAB_SIZE:
        return cfg
    return ReasonerConfig.from_dict({**cfg.to_dict(), "vocab_size": BYTE_VOCAB_SIZE})


def count_parameters(core: ReasonerCore) -> int:
    return int(sum(parameter.numel() for parameter in core.parameters()))


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


def stable_row_hash(text: str, seed: int = 1337) -> int:
    payload = f"{seed}\0{text}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def split_for_text(text: str, spec: CSVSplitSpec) -> str:
    spec.validate()
    bucket = stable_row_hash(text, spec.seed) / float(2**64 - 1)
    if bucket < spec.train:
        return "train"
    if bucket < spec.train + spec.val:
        return "val"
    return "test"


def iter_language_texts(
    path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
) -> Iterator[str]:
    yield from iter_language_examples(path, text_column, target_column, max_rows=max_rows, split="all")


def iter_language_examples(
    path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
    split: SplitName = "all",
    split_spec: CSVSplitSpec | None = None,
) -> Iterator[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train, val, test, or all.")

    header = _sniff_header(csv_path)
    text_col, target_col = detect_language_columns(header, text_column, target_column)
    spec = split_spec or CSVSplitSpec()
    if split != "all":
        spec.validate()

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
            if split != "all" and split_for_text(text, spec) != split:
                continue
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


def audit_language_csv(
    path: str | Path,
    text_column: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
    split_spec: CSVSplitSpec | None = None,
) -> CSVCorpusAudit:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    spec = split_spec or CSVSplitSpec()
    spec.validate()
    header = _sniff_header(csv_path)
    text_col, target_col = detect_language_columns(header, text_column, target_column)

    rows_seen = 0
    nonempty = 0
    total_chars = 0
    total_bytes = 0
    min_chars: int | None = None
    max_chars = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    seen_hashes: set[int] = set()
    duplicates = 0

    for text in iter_language_texts(csv_path, text_col, target_col, max_rows=max_rows):
        rows_seen += 1
        if not text:
            continue
        nonempty += 1
        length = len(text)
        byte_length = len(text.encode("utf-8", errors="replace"))
        total_chars += length
        total_bytes += byte_length
        min_chars = length if min_chars is None else min(min_chars, length)
        max_chars = max(max_chars, length)
        split_name = split_for_text(text, spec)
        split_counts[split_name] += 1
        row_hash = stable_row_hash(text, seed=0)
        if row_hash in seen_hashes:
            duplicates += 1
        else:
            seen_hashes.add(row_hash)

    warnings: list[str] = []
    if nonempty == 0:
        warnings.append("no nonempty training rows found")
    if split_counts["val"] == 0:
        warnings.append("validation split is empty under current max_rows/split; increase max_rows or val fraction")
    if split_counts["test"] == 0:
        warnings.append("test split is empty under current max_rows/split; increase max_rows or test fraction")
    if duplicates and nonempty:
        warnings.append("duplicate rows detected; consider deduplicating before larger training")
    if text_column is None and text_col == header[0] and text_col.lower() not in {*_TEXT_COLUMNS, *_PROMPT_COLUMNS}:
        warnings.append("text column was inferred from first CSV column; pass --text-column for reproducibility")

    return CSVCorpusAudit(
        path=str(csv_path),
        header=header,
        text_column=text_col,
        target_column=target_col,
        max_rows=max_rows,
        rows_seen=rows_seen,
        nonempty_rows=nonempty,
        duplicate_rows=duplicates,
        duplicate_rate=float(duplicates / nonempty) if nonempty else 0.0,
        split_counts=split_counts,
        min_chars=min_chars or 0,
        max_chars=max_chars,
        mean_chars=float(total_chars / nonempty) if nonempty else 0.0,
        total_chars=total_chars,
        total_utf8_bytes=total_bytes,
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
    split_spec: CSVSplitSpec,
) -> Iterator[str]:
    while True:
        yielded = False
        for text in iter_language_examples(path, text_column, target_column, max_rows=max_rows, split="train", split_spec=split_spec):
            yielded = True
            yield text
        if not yielded:
            raise ValueError("CSV train split did not yield any nonempty rows. Increase max_rows or train split.")


def _split_sample(
    path: Path,
    text_column: str,
    target_column: str | None,
    max_rows: int | None,
    split: SplitName,
    split_spec: CSVSplitSpec,
    limit: int,
) -> list[str]:
    return list(islice(iter_language_examples(path, text_column, target_column, max_rows=max_rows, split=split, split_spec=split_spec), limit))


def _encoded_targets(texts: list[str], tokenizer: ByteTokenizer, seq_len: int) -> list[int]:
    tokens: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, seq_len + 1)[1:]  # next-token targets after BOS
        tokens.extend(token for token in encoded if token != tokenizer.pad_id)
    return tokens


def byte_unigram_baseline(
    train_rows: list[str],
    eval_rows: list[str],
    tokenizer: ByteTokenizer,
    seq_len: int,
    alpha: float = 1.0,
) -> ByteUnigramBaseline:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    counts = torch.full((tokenizer.vocab_size,), float(alpha), dtype=torch.float64)
    for token in _encoded_targets(train_rows, tokenizer, seq_len):
        counts[int(token)] += 1.0
    probs = counts / counts.sum().clamp_min(1.0)
    eval_tokens = _encoded_targets(eval_rows, tokenizer, seq_len)
    if not eval_tokens:
        return ByteUnigramBaseline(len(train_rows), len(eval_rows), 0, 0.0, 0.0)
    nll = -sum(float(probs[int(token)].log().item()) for token in eval_tokens) / len(eval_tokens)
    return ByteUnigramBaseline(
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        eval_tokens=len(eval_tokens),
        loss=float(nll),
        perplexity=float(math.exp(min(nll, 20.0))),
    )


def _write_jsonl(path: Path, rows: list[CSVTrainingStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def _save_checkpoint(
    checkpoint: Path,
    core: ReasonerCore,
    optimizer: torch.optim.Optimizer,
    language_cfg: ReasonerConfig,
    tokenizer: ByteTokenizer,
    training: dict[str, Any],
    step: int,
    eval_loss: float,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": core.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": language_cfg.to_dict(),
            "tokenizer": {
                "type": "byte-level",
                "pad_id": tokenizer.pad_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
                "byte_offset": BYTE_OFFSET,
                "vocab_size": tokenizer.vocab_size,
            },
            "training": {**training, "step": step, "eval_loss": eval_loss},
        },
        checkpoint,
    )


def _load_checkpoint(path: str | Path, core: ReasonerCore, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(Path(path), map_location=core.device_obj)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint missing model_state_dict")
    core.load_state_dict(state_dict)
    if optimizer is not None and isinstance(checkpoint.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    training = checkpoint.get("training") if isinstance(checkpoint.get("training"), dict) else {}
    return int(training.get("step", 0) or 0)


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
    log_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    eval_interval: int = 5,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    split_seed: int | None = None,
    baseline_rows: int = 256,
) -> CSVLanguageProbeResult:
    """Run a held-out CSV language-readiness experiment.

    This is the real v0.4 training layer: streaming CSV ingestion,
    deterministic split assignment, dataset audit, unigram baseline, held-out
    eval, checkpoint/resume, and JSONL training logs. It remains a tiny
    architecture probe, not full language pretraining.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if eval_rows <= 0:
        raise ValueError("eval_rows must be positive.")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if baseline_rows <= 0:
        raise ValueError("baseline_rows must be positive.")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    path = Path(csv_path)
    split_spec = CSVSplitSpec(train_fraction, val_fraction, test_fraction, split_seed if split_seed is not None else cfg.seed)
    split_spec.validate()
    audit = audit_language_csv(path, text_column, target_column, max_rows=max_rows, split_spec=split_spec)
    inspection = inspect_language_csv(path, audit.text_column, audit.target_column, sample_rows=min(max_rows or 1000, 1000))
    if audit.nonempty_rows == 0:
        raise ValueError("CSV has no nonempty language rows.")
    if audit.split_counts["train"] == 0:
        raise ValueError("train split is empty. Increase max_rows or train fraction.")

    language_cfg = language_ready_config(cfg)
    safe_seq_len = min(seq_len, language_cfg.max_seq_len)
    set_seed(language_cfg.seed)
    core = ReasonerCore(language_cfg)
    tokenizer = ByteTokenizer()
    opt = torch.optim.AdamW(core.parameters(), lr=lr)
    resumed_step = 0
    if resume_from is not None:
        resumed_step = _load_checkpoint(resume_from, core, opt)

    val_sample = _split_sample(path, audit.text_column, audit.target_column, max_rows, "val", split_spec, eval_rows)
    if not val_sample:
        val_sample = _split_sample(path, audit.text_column, audit.target_column, max_rows, "train", split_spec, eval_rows)
    train_baseline_sample = _split_sample(path, audit.text_column, audit.target_column, max_rows, "train", split_spec, baseline_rows)
    baseline = byte_unigram_baseline(train_baseline_sample, val_sample, tokenizer, safe_seq_len)
    uniform_baseline_loss = float(math.log(tokenizer.vocab_size))
    initial_eval_loss, initial_eval_acc = _evaluate(core, val_sample, tokenizer, safe_seq_len)

    iterator = _make_training_iterator(path, audit.text_column, audit.target_column, max_rows, split_spec)
    final_loss = 0.0
    rows_consumed = 0
    tokens_seen = 0
    history: list[CSVTrainingStep] = []
    start = time.perf_counter()
    best_eval_loss = initial_eval_loss
    best_checkpoint: Path | None = None
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        best_checkpoint = checkpoint.with_name(f"{checkpoint.stem}.best{checkpoint.suffix or '.pt'}")

    core.train()
    for step in range(1, steps + 1):
        batch = _next_text_batch(iterator, batch_size)
        if len(batch) < batch_size:
            iterator = _make_training_iterator(path, audit.text_column, audit.target_column, max_rows, split_spec)
            batch.extend(_next_text_batch(iterator, batch_size - len(batch)))
        if not batch:
            raise ValueError("CSV train iterator produced no batch")
        rows_consumed += len(batch)
        x, y = _batch_from_texts(batch, tokenizer, safe_seq_len, core.device_obj)
        tokens_seen += int(y.ne(-100).sum().item())
        opt.zero_grad(set_to_none=True)
        logits = core(x, mode="think" if language_cfg.recurrent_depth > 1 else "fast")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.item())

        should_eval = step == 1 or step == steps or step % eval_interval == 0
        eval_loss: float | None = None
        eval_acc: float | None = None
        if should_eval:
            eval_loss, eval_acc = _evaluate(core, val_sample, tokenizer, safe_seq_len)
            core.train()
            if best_checkpoint is not None and eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                _save_checkpoint(
                    best_checkpoint,
                    core,
                    opt,
                    language_cfg,
                    tokenizer,
                    {
                        "csv_path": str(path),
                        "text_column": audit.text_column,
                        "target_column": audit.target_column,
                        "steps": steps,
                        "batch_size": batch_size,
                        "seq_len": safe_seq_len,
                        "max_rows": max_rows,
                        "lr": lr,
                        "split_spec": split_spec.to_dict(),
                    },
                    resumed_step + step,
                    eval_loss,
                )
        history.append(
            CSVTrainingStep(
                step=resumed_step + step,
                train_loss=final_loss,
                eval_loss=eval_loss,
                eval_token_accuracy=eval_acc,
                seconds_elapsed=float(time.perf_counter() - start),
            )
        )

    eval_loss, eval_acc = _evaluate(core, val_sample, tokenizer, safe_seq_len)
    checkpoint_out: str | None = None
    if checkpoint is not None:
        _save_checkpoint(
            checkpoint,
            core,
            opt,
            language_cfg,
            tokenizer,
            {
                "csv_path": str(path),
                "text_column": audit.text_column,
                "target_column": audit.target_column,
                "steps": steps,
                "batch_size": batch_size,
                "seq_len": safe_seq_len,
                "max_rows": max_rows,
                "lr": lr,
                "split_spec": split_spec.to_dict(),
            },
            resumed_step + steps,
            eval_loss,
        )
        checkpoint_out = str(checkpoint)

    log_out: str | None = None
    if log_path is not None:
        log = Path(log_path)
        _write_jsonl(log, history)
        log_out = str(log)

    improvement_vs_initial = initial_eval_loss - eval_loss
    improvement_vs_unigram = (baseline.loss - eval_loss) if baseline.loss > 0 else None
    recommendation = _csv_training_recommendation(
        eval_loss=eval_loss,
        initial_loss=initial_eval_loss,
        unigram_loss=baseline.loss,
        audit=audit,
    )

    return CSVLanguageProbeResult(
        csv_path=str(path),
        text_column=audit.text_column,
        target_column=audit.target_column,
        steps=steps,
        batch_size=batch_size,
        seq_len=safe_seq_len,
        max_rows=max_rows,
        rows_consumed=rows_consumed,
        initial_loss=float(initial_eval_loss),
        final_loss=float(final_loss),
        loss_delta=float(improvement_vs_initial),
        eval_loss=float(eval_loss),
        eval_token_accuracy=float(eval_acc),
        byte_level_perplexity=float(math.exp(min(eval_loss, 20.0))) if eval_loss > 0 else 0.0,
        checkpoint_path=checkpoint_out,
        inspection=inspection.to_dict(),
        audit=audit.to_dict(),
        split_spec=split_spec.to_dict(),
        uniform_baseline_loss=uniform_baseline_loss,
        unigram_baseline=baseline.to_dict(),
        best_eval_loss=float(best_eval_loss),
        best_checkpoint_path=str(best_checkpoint) if best_checkpoint is not None and best_checkpoint.exists() else None,
        log_path=log_out,
        history=[row.to_dict() for row in history],
        model_parameters=count_parameters(core),
        tokens_seen=tokens_seen,
        resumed_from=str(resume_from) if resume_from is not None else None,
        recommendation=recommendation,
    )


def _csv_training_recommendation(
    eval_loss: float,
    initial_loss: float,
    unigram_loss: float,
    audit: CSVCorpusAudit,
) -> str:
    if audit.split_counts["val"] == 0 or audit.nonempty_rows < 20:
        return "RE-TEST — dataset is too small or validation split is empty; this run is only a smoke test."
    improved_initial = eval_loss < initial_loss
    beats_unigram = unigram_loss > 0 and eval_loss < unigram_loss
    if improved_initial and beats_unigram:
        return "KEEP — model beat initial random loss and byte-unigram baseline on held-out validation."
    if improved_initial:
        return "RE-TEST — model learned from random initialization but did not beat the byte-unigram baseline."
    return "KILL/REWORK — held-out loss did not improve under this budget."


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
    log_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    eval_interval: int = 5,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    split_seed: int | None = None,
    baseline_rows: int = 256,
) -> BenchmarkReport:
    """Return a standard WAI-R0 report for a CSV language-readiness run."""

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
        log_path=log_path,
        resume_from=resume_from,
        eval_interval=eval_interval,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        baseline_rows=baseline_rows,
    )
    return BenchmarkReport(
        name="csv_language_readiness",
        result_type="v0.4 CSV language-readiness experiment",
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
            "log_path": str(log_path) if log_path is not None else None,
            "resume_from": str(resume_from) if resume_from is not None else None,
            "eval_interval": eval_interval,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "baseline_rows": baseline_rows,
        },
        raw_metrics=result.to_dict(),
        summary=(
            "CSV language-readiness experiment completed. The run audits the corpus, assigns stable "
            "train/validation/test splits, compares against byte-level baselines, trains a tiny causal "
            "probe, and reports held-out validation loss. This is a real training/eval harness, not a "
            "claim of semantic reasoning."
        ),
        limitations=[
            "The tokenizer is byte-level and dependency-free; serious language work needs a better tokenizer.",
            "The held-out split is hash-based by row text, not document-aware; duplicated templates can still leak structure.",
            "The model is intentionally tiny for local probing; success here only justifies a larger controlled run.",
            "This does not evaluate instruction following, factuality, safety, or general reasoning.",
        ],
        recommendation=result.recommendation,
    )


@torch.no_grad()
def generate_from_csv_checkpoint(
    checkpoint_path: str | Path,
    prompt: str = "",
    max_new_tokens: int = 64,
) -> str:
    """Greedy byte-level sample from a WAI-R0 CSV checkpoint.

    This is intentionally deterministic and boring. It is an inspection utility,
    not a chat interface and not evidence of conversational ability.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    config_data = checkpoint.get("model_config")
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint missing model_config")
    cfg = language_ready_config(ReasonerConfig.from_dict(config_data))
    core = ReasonerCore(cfg)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")
    core.load_state_dict(state)
    core.eval()
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(prompt, min(max(4, cfg.max_seq_len), max(4, len(prompt.encode("utf-8")) + 2)))
    tokens = torch.tensor([ids], dtype=torch.long, device=core.device_obj)
    out = core.transformer.generate(tokens, max_new_tokens=max_new_tokens)[0].tolist()
    return tokenizer.decode(out)
