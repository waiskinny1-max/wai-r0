from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from wai_r0.core.reproducibility import file_sha256
from wai_r0.data.dedupe import DuplicateIndex
from wai_r0.data.manifest import DatasetManifest, LengthSummary
from wai_r0.data.schema import (
    CANONICAL_FIELDS,
    REQUIRED_FIELDS,
    ConversationRow,
)
from wai_r0.data.splits import SplitSpec, assign_split

_HEADER_ALIASES = {
    "prompt": "user",
    "instruction": "user",
    "response": "assistant",
    "answer": "assistant",
    "completion": "assistant",
    "metadata": "metadata_json",
    "family": "task_family",
}


@dataclass(slots=True)
class DatasetAudit:
    source: str
    sha256: str
    source_bytes: int = 0
    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    duplicate_ids: int = 0
    exact_duplicate_content_rows: int = 0
    near_duplicate_content_rows: int = 0
    cross_split_duplicate_rows: int = 0
    empty_user_rows: int = 0
    empty_assistant_rows: int = 0
    invalid_metadata_rows: int = 0
    oversized_field_rows: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    declared_split_counts: dict[str, int] = field(default_factory=dict)
    task_family_counts: dict[str, int] = field(default_factory=dict)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    length_summaries: dict[str, LengthSummary] = field(default_factory=dict)
    rejection_examples: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_manifest(self, *, split_spec: SplitSpec) -> DatasetManifest:
        limitations = [
            "Near-duplicate detection uses SimHash and can produce false positives or misses.",
            "Character lengths are not tokenizer-specific token counts.",
        ]
        if self.rejected_rows:
            limitations.append("Rejected rows are excluded from split and distribution counts.")
        return DatasetManifest(
            source=self.source,
            source_sha256=self.sha256,
            source_bytes=self.source_bytes,
            schema_fields=list(CANONICAL_FIELDS),
            accepted_rows=self.accepted_rows,
            rejected_rows=self.rejected_rows,
            split_counts=dict(self.split_counts),
            task_family_counts=dict(self.task_family_counts),
            difficulty_counts=dict(self.difficulty_counts),
            exact_duplicate_content_rows=self.exact_duplicate_content_rows,
            near_duplicate_content_rows=self.near_duplicate_content_rows,
            length_summaries=dict(self.length_summaries),
            split_policy=split_spec.to_dict(),
            limitations=limitations,
        )


class _ReservoirLengths:
    def __init__(self, *, capacity: int = 4096, seed: int = 0) -> None:
        self.capacity = capacity
        self.random = random.Random(seed)
        self.count = 0
        self.total = 0
        self.minimum: int | None = None
        self.maximum: int | None = None
        self.sample: list[int] = []

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.sample) < self.capacity:
            self.sample.append(value)
        else:
            index = self.random.randrange(self.count)
            if index < self.capacity:
                self.sample[index] = value

    def summary(self) -> LengthSummary:
        if not self.count:
            return LengthSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        ordered = sorted(self.sample)

        def quantile(q: float) -> float:
            if len(ordered) == 1:
                return float(ordered[0])
            position = q * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            fraction = position - lower
            return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

        return LengthSummary(
            count=self.count,
            minimum=int(self.minimum or 0),
            maximum=int(self.maximum or 0),
            mean=self.total / self.count,
            p50=quantile(0.50),
            p90=quantile(0.90),
            p99=quantile(0.99),
        )


def _canonical_header(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError("CSV is missing a header row")
    mapping: dict[str, str] = {}
    for original in fieldnames:
        if original is None:
            continue
        normalized = original.strip().casefold().replace(" ", "_")
        canonical = _HEADER_ALIASES.get(normalized, normalized)
        if canonical in mapping.values():
            raise ValueError(f"CSV has multiple columns resolving to {canonical!r}")
        mapping[original] = canonical
    missing = sorted(set(REQUIRED_FIELDS) - set(mapping.values()))
    if missing:
        raise ValueError(f"CSV is missing required fields: {', '.join(missing)}")
    return mapping


def _normalized_row(raw: dict[str | None, str | None], header: dict[str, str]) -> dict[str, str]:
    row: dict[str, str] = {}
    for original, value in raw.items():
        if original is None:
            continue
        canonical = header.get(original)
        if canonical is not None:
            row[canonical] = (value or "").strip()
    return row


def _parse_metadata(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must decode to an object")
    return parsed


def _conversation_from_mapping(
    row: dict[str, str],
    *,
    line_number: int,
    max_field_chars: int,
) -> ConversationRow:
    try:
        metadata = _parse_metadata(row.get("metadata_json", ""))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid metadata_json: {exc}") from exc
    conversation = ConversationRow(
        id=row.get("id", ""),
        split=row.get("split", ""),
        task_family=row.get("task_family", ""),
        difficulty=row.get("difficulty", ""),
        system=row.get("system", ""),
        user=row.get("user", ""),
        assistant=row.get("assistant", ""),
        answer_format=row.get("answer_format", ""),
        eval_type=row.get("eval_type", ""),
        metadata=metadata,
        source_line=line_number,
    )
    conversation.validate(max_field_chars=max_field_chars)
    return conversation


def iter_conversation_rows(
    path: str | Path,
    *,
    max_rows: int | None = None,
    max_field_chars: int = 1_000_000,
) -> Iterator[ConversationRow]:
    """Stream valid canonical rows and fail closed on the first malformed row."""

    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows cannot be negative")
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = _canonical_header(reader.fieldnames)
        for emitted, raw in enumerate(reader):
            if max_rows is not None and emitted >= max_rows:
                break
            line_number = emitted + 2
            row = _normalized_row(raw, header)
            try:
                conversation = _conversation_from_mapping(
                    row, line_number=line_number, max_field_chars=max_field_chars
                )
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            yield conversation


def audit_conversation_csv(
    path: str | Path,
    *,
    rejection_sample_limit: int = 20,
    max_rows: int | None = None,
    max_field_chars: int = 1_000_000,
    split_spec: SplitSpec | None = None,
    near_duplicate_distance: int = 3,
) -> DatasetAudit:
    """Audit a canonical conversation CSV in one bounded-memory pass."""

    if rejection_sample_limit < 0:
        raise ValueError("rejection_sample_limit cannot be negative")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows cannot be negative")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = split_spec or SplitSpec()
    spec.validate()
    audit = DatasetAudit(
        source=str(source),
        sha256=file_sha256(source),
        source_bytes=source.stat().st_size,
    )
    split_counts: Counter[str] = Counter()
    declared_split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_index = DuplicateIndex(near_distance=near_duplicate_distance)
    content_splits: dict[str, str] = {}
    lengths = {
        "system_chars": _ReservoirLengths(seed=1),
        "user_chars": _ReservoirLengths(seed=2),
        "assistant_chars": _ReservoirLengths(seed=3),
        "combined_chars": _ReservoirLengths(seed=4),
    }

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = _canonical_header(reader.fieldnames)
        for line_number, raw in enumerate(reader, start=2):
            if max_rows is not None and audit.total_rows >= max_rows:
                break
            audit.total_rows += 1
            row = _normalized_row(raw, header)
            reasons: list[str] = []
            if not row.get("user"):
                audit.empty_user_rows += 1
                reasons.append("empty_user")
            if not row.get("assistant"):
                audit.empty_assistant_rows += 1
                reasons.append("empty_assistant")
            if any(
                len(row.get(name, "")) > max_field_chars for name in ("system", "user", "assistant")
            ):
                audit.oversized_field_rows += 1
                reasons.append("oversized_field")
            try:
                metadata = _parse_metadata(row.get("metadata_json", ""))
            except (json.JSONDecodeError, ValueError):
                metadata = {}
                audit.invalid_metadata_rows += 1
                reasons.append("invalid_metadata_json")

            row_id = row.get("id", "")
            if row_id:
                if row_id in seen_ids:
                    audit.duplicate_ids += 1
                    reasons.append("duplicate_id")
                seen_ids.add(row_id)

            if reasons:
                audit.rejected_rows += 1
                if len(audit.rejection_examples) < rejection_sample_limit:
                    audit.rejection_examples.append(
                        {"line": line_number, "id": row_id or None, "reasons": reasons}
                    )
                continue

            conversation = ConversationRow(
                id=row_id,
                split=row.get("split", ""),
                task_family=row.get("task_family", ""),
                difficulty=row.get("difficulty", ""),
                system=row.get("system", ""),
                user=row["user"],
                assistant=row["assistant"],
                answer_format=row.get("answer_format", ""),
                eval_type=row.get("eval_type", ""),
                metadata=metadata,
                source_line=line_number,
            )
            assigned_split = assign_split(conversation, spec)
            match = duplicate_index.add(conversation.normalized_content)
            if match.exact:
                audit.exact_duplicate_content_rows += 1
            elif match.near:
                audit.near_duplicate_content_rows += 1
            previous_split = content_splits.setdefault(conversation.content_hash, assigned_split)
            if previous_split != assigned_split:
                audit.cross_split_duplicate_rows += 1

            audit.accepted_rows += 1
            split_counts[assigned_split] += 1
            declared_split_counts[conversation.normalized_split or "unspecified"] += 1
            family_counts[conversation.task_family or "unspecified"] += 1
            difficulty_counts[conversation.difficulty or "unspecified"] += 1
            lengths["system_chars"].add(len(conversation.system))
            lengths["user_chars"].add(len(conversation.user))
            lengths["assistant_chars"].add(len(conversation.assistant))
            lengths["combined_chars"].add(
                len(conversation.system) + len(conversation.user) + len(conversation.assistant)
            )

    audit.split_counts = dict(sorted(split_counts.items()))
    audit.declared_split_counts = dict(sorted(declared_split_counts.items()))
    audit.task_family_counts = dict(sorted(family_counts.items()))
    audit.difficulty_counts = dict(sorted(difficulty_counts.items()))
    audit.length_summaries = {name: tracker.summary() for name, tracker in lengths.items()}
    if audit.cross_split_duplicate_rows:
        audit.warnings.append("Exact duplicate content crosses assigned splits.")
    if spec.respect_declared and not {"val", "test"}.intersection(audit.split_counts):
        audit.warnings.append("Declared split mode produced no validation or test rows.")
    if audit.accepted_rows == 0:
        audit.warnings.append("No valid training rows were accepted.")
    return audit


__all__ = [
    "CANONICAL_FIELDS",
    "REQUIRED_FIELDS",
    "ConversationRow",
    "DatasetAudit",
    "audit_conversation_csv",
    "iter_conversation_rows",
]
