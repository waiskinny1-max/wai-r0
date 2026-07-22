from __future__ import annotations

import json
import math
import mmap
import os
import random
import shutil
import struct
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from wai_r0.core.reproducibility import (
    _fsync_directory,
    atomic_write_json,
    canonical_hash,
    file_sha256,
)
from wai_r0.data.chat import EncodedChatExample, Tokenizer, encode_chat_example, pad_chat_batch
from wai_r0.data.csv_reader import audit_conversation_csv, iter_conversation_rows
from wai_r0.data.packing import pack_chat_examples
from wai_r0.data.splits import SplitSpec, assign_split

COMPILED_DATASET_FORMAT_VERSION = 2
_SUPPORTED_COMPILED_DATASET_FORMATS = {1, 2}
_ENTRY = struct.Struct("<QQQQ")
_INT32 = struct.Struct("<i")
SplitLiteral = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class CompiledSplitSummary:
    split: SplitLiteral
    examples: int
    input_tokens: int
    target_tokens: int
    truncated_examples: int
    tokens_file: str
    labels_file: str
    index_file: str
    tokens_sha256: str
    labels_sha256: str
    index_sha256: str
    raw_utf8_bytes: int = 0
    target_utf8_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompiledDatasetManifest:
    format_version: int
    source: str
    source_sha256: str
    tokenizer_manifest: dict[str, Any]
    tokenizer_manifest_hash: str
    chat_template_hash: str
    split_spec: dict[str, Any]
    max_length: int
    assistant_only_loss: bool
    audit: dict[str, Any]
    splits: dict[str, CompiledSplitSummary]

    def validate(self) -> None:
        if self.format_version not in _SUPPORTED_COMPILED_DATASET_FORMATS:
            raise ValueError("unsupported compiled dataset format")
        if len(self.source_sha256) != 64:
            raise ValueError("invalid source SHA-256")
        actual_tokenizer_hash = self.tokenizer_manifest.get("manifest_hash")
        if actual_tokenizer_hash != self.tokenizer_manifest_hash:
            raise ValueError("tokenizer manifest hash mismatch")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if not self.splits or "train" not in self.splits:
            raise ValueError("compiled dataset must contain a training split")
        if self.splits["train"].examples < 1:
            raise ValueError("compiled training split is empty")
        if self.format_version >= 2:
            if self.audit.get("sha256") != self.source_sha256:
                raise ValueError("compiled dataset audit source hash mismatch")
            compiled_examples = sum(summary.examples for summary in self.splits.values())
            if int(self.audit.get("accepted_rows", -1)) != compiled_examples:
                raise ValueError("compiled dataset audit row count mismatch")
            if int(self.audit.get("rejected_rows", 0)) != 0:
                raise ValueError("compiled dataset audit contains rejected rows")
        for name, summary in self.splits.items():
            if name != summary.split:
                raise ValueError("compiled split mapping key does not match summary")
            if (
                min(
                    summary.examples,
                    summary.input_tokens,
                    summary.target_tokens,
                    summary.truncated_examples,
                    summary.raw_utf8_bytes,
                    summary.target_utf8_bytes,
                )
                < 0
            ):
                raise ValueError("compiled split counters cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        body: dict[str, Any] = {
            "format_version": self.format_version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "tokenizer_manifest": self.tokenizer_manifest,
            "tokenizer_manifest_hash": self.tokenizer_manifest_hash,
            "chat_template_hash": self.chat_template_hash,
            "split_spec": self.split_spec,
            "max_length": self.max_length,
            "assistant_only_loss": self.assistant_only_loss,
            "splits": {name: summary.to_dict() for name, summary in sorted(self.splits.items())},
        }
        if self.format_version >= 2:
            body["audit"] = self.audit
        body["manifest_hash"] = canonical_hash(body)
        return body

    @classmethod
    def load(cls, path: str | Path) -> CompiledDatasetManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("compiled dataset manifest root must be an object")
        expected_hash = payload.pop("manifest_hash", None)
        if expected_hash is not None and expected_hash != canonical_hash(payload):
            raise ValueError("compiled dataset manifest hash mismatch")
        raw_splits = payload.get("splits")
        if not isinstance(raw_splits, dict):
            raise ValueError("compiled dataset manifest is missing splits")
        splits = {
            str(name): CompiledSplitSummary(**dict(summary))
            for name, summary in raw_splits.items()
            if isinstance(summary, Mapping)
        }
        if len(splits) != len(raw_splits):
            raise ValueError("compiled split summary must be an object")
        manifest = cls(
            format_version=int(payload.get("format_version", 0)),
            source=str(payload.get("source", "")),
            source_sha256=str(payload.get("source_sha256", "")),
            tokenizer_manifest=dict(payload.get("tokenizer_manifest", {})),
            tokenizer_manifest_hash=str(payload.get("tokenizer_manifest_hash", "")),
            chat_template_hash=str(payload.get("chat_template_hash", "")),
            split_spec=dict(payload.get("split_spec", {})),
            max_length=int(payload.get("max_length", 0)),
            assistant_only_loss=bool(payload.get("assistant_only_loss", True)),
            audit=dict(payload.get("audit", {})),
            splits=splits,
        )
        manifest.validate()
        return manifest


@dataclass(slots=True)
class _WriterState:
    split: SplitLiteral
    tokens_handle: Any
    labels_handle: Any
    index_handle: Any
    tokens_path: Path
    labels_path: Path
    index_path: Path
    examples: int = 0
    input_tokens: int = 0
    target_tokens: int = 0
    truncated_examples: int = 0
    raw_utf8_bytes: int = 0
    target_utf8_bytes: int = 0

    def append(
        self,
        example: EncodedChatExample,
        source_line: int | None,
        raw_utf8_bytes: int,
        target_utf8_bytes: int,
    ) -> None:
        input_ids = [int(value) for value in example.input_ids.tolist()]
        labels = [int(value) for value in example.labels.tolist()]
        if len(input_ids) != len(labels) or not input_ids:
            raise ValueError("encoded example must contain aligned non-empty input and labels")
        offset = self.input_tokens
        self.tokens_handle.write(struct.pack(f"<{len(input_ids)}i", *input_ids))
        self.labels_handle.write(struct.pack(f"<{len(labels)}i", *labels))
        self.index_handle.write(
            _ENTRY.pack(
                offset,
                len(input_ids),
                int(example.target_token_count),
                max(0, int(source_line or 0)),
            )
        )
        self.examples += 1
        self.input_tokens += len(input_ids)
        self.target_tokens += int(example.target_token_count)
        self.truncated_examples += int(example.truncated)
        self.raw_utf8_bytes += int(raw_utf8_bytes)
        self.target_utf8_bytes += int(target_utf8_bytes)

    def close(self) -> None:
        for handle in (self.tokens_handle, self.labels_handle, self.index_handle):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def summary(self, root: Path) -> CompiledSplitSummary:
        return CompiledSplitSummary(
            split=self.split,
            examples=self.examples,
            input_tokens=self.input_tokens,
            target_tokens=self.target_tokens,
            truncated_examples=self.truncated_examples,
            tokens_file=str(self.tokens_path.relative_to(root)),
            labels_file=str(self.labels_path.relative_to(root)),
            index_file=str(self.index_path.relative_to(root)),
            tokens_sha256=file_sha256(self.tokens_path),
            labels_sha256=file_sha256(self.labels_path),
            index_sha256=file_sha256(self.index_path),
            raw_utf8_bytes=self.raw_utf8_bytes,
            target_utf8_bytes=self.target_utf8_bytes,
        )


def _open_writer(root: Path, split: SplitLiteral) -> _WriterState:
    shard = root / "shards"
    shard.mkdir(parents=True, exist_ok=True)
    prefix = shard / f"{split}-00000"
    tokens_path = prefix.with_suffix(".tokens.bin")
    labels_path = prefix.with_suffix(".labels.bin")
    index_path = prefix.with_suffix(".index.bin")
    return _WriterState(
        split=split,
        tokens_handle=tokens_path.open("wb"),
        labels_handle=labels_path.open("wb"),
        index_handle=index_path.open("wb"),
        tokens_path=tokens_path,
        labels_path=labels_path,
        index_path=index_path,
    )


def compile_conversation_dataset(
    csv_path: str | Path,
    *,
    output_dir: str | Path,
    tokenizer: Tokenizer,
    split_spec: SplitSpec | None = None,
    max_length: int = 512,
    assistant_only_loss: bool = True,
    max_rows: int | None = None,
    overwrite: bool = False,
    allow_cross_split_duplicates: bool = False,
) -> Path:
    """Compile canonical CSV rows into checksum-verified memory-mappable shards."""

    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"compiled dataset already exists: {destination}")
    spec = split_spec or SplitSpec()
    spec.validate()
    audit = audit_conversation_csv(source, max_rows=max_rows, split_spec=spec)
    if audit.rejected_rows:
        raise ValueError(
            f"dataset audit rejected {audit.rejected_rows} rows; fix the source before compiling"
        )
    if audit.cross_split_duplicate_rows and not allow_cross_split_duplicates:
        raise ValueError(
            f"dataset audit found {audit.cross_split_duplicate_rows} exact cross-split "
            "duplicate rows; fix the split or pass the explicit override"
        )
    if audit.split_counts.get("train", 0) < 1:
        raise ValueError("dataset audit found no training rows")
    tokenizer_manifest = tokenizer.manifest()
    tokenizer_hash = tokenizer_manifest.get("manifest_hash")
    if not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64:
        raise ValueError("tokenizer manifest must contain a valid manifest_hash")
    chat_template_hash = canonical_hash(
        {
            "version": tokenizer_manifest.get("chat_template_version", 1),
            "roles": ["system", "user", "assistant"],
            "assistant_only_loss": assistant_only_loss,
            "target_truncation": "preserve_assistant_then_left_truncate_context",
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )
    writers: dict[SplitLiteral, _WriterState] = {
        split: _open_writer(temporary, split) for split in ("train", "val", "test")
    }
    try:
        for row in iter_conversation_rows(source, max_rows=max_rows):
            split = assign_split(row, spec)
            encoded = encode_chat_example(
                row.chat,
                tokenizer=tokenizer,
                max_length=max_length,
                assistant_only_loss=assistant_only_loss,
            )
            raw_utf8_bytes = sum(
                len(value.encode("utf-8")) for value in (row.system, row.user, row.assistant)
            )
            target_utf8_bytes = (
                len(row.assistant.encode("utf-8")) if assistant_only_loss else raw_utf8_bytes
            )
            writers[split].append(encoded, row.source_line, raw_utf8_bytes, target_utf8_bytes)
        for writer in writers.values():
            writer.close()
        summaries = {str(name): writer.summary(temporary) for name, writer in writers.items()}
        manifest = CompiledDatasetManifest(
            format_version=COMPILED_DATASET_FORMAT_VERSION,
            source=str(source),
            source_sha256=file_sha256(source),
            tokenizer_manifest=tokenizer_manifest,
            tokenizer_manifest_hash=tokenizer_hash,
            chat_template_hash=chat_template_hash,
            split_spec=spec.to_dict(),
            max_length=max_length,
            assistant_only_loss=assistant_only_loss,
            audit=audit.to_dict(),
            splits=summaries,
        )
        atomic_write_json(temporary / "manifest.json", manifest.to_dict())
        _fsync_directory(temporary / "shards")
        _fsync_directory(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        for writer in writers.values():
            for handle in (writer.tokens_handle, writer.labels_handle, writer.index_handle):
                if not handle.closed:
                    handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "manifest.json"


def verify_compiled_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root if root.name == "manifest.json" else root / "manifest.json"
    manifest = CompiledDatasetManifest.load(manifest_path)
    base = manifest_path.parent
    failures: list[str] = []
    for summary in manifest.splits.values():
        for relative, expected in (
            (summary.tokens_file, summary.tokens_sha256),
            (summary.labels_file, summary.labels_sha256),
            (summary.index_file, summary.index_sha256),
        ):
            candidate = base / relative
            if not candidate.is_file():
                failures.append(f"missing:{relative}")
            elif file_sha256(candidate) != expected:
                failures.append(f"digest:{relative}")
        index_path = base / summary.index_file
        if index_path.is_file() and index_path.stat().st_size != summary.examples * _ENTRY.size:
            failures.append(f"index_size:{summary.index_file}")
        tokens_path = base / summary.tokens_file
        labels_path = base / summary.labels_file
        expected_bytes = summary.input_tokens * _INT32.size
        if tokens_path.is_file() and tokens_path.stat().st_size != expected_bytes:
            failures.append(f"token_size:{summary.tokens_file}")
        if labels_path.is_file() and labels_path.stat().st_size != expected_bytes:
            failures.append(f"label_size:{summary.labels_file}")
    source_path = Path(manifest.source)
    source_available = source_path.is_file()
    if source_available and file_sha256(source_path) != manifest.source_sha256:
        failures.append("source_digest")
    return {
        "valid": not failures,
        "manifest": manifest.to_dict(),
        "failures": failures,
        "source_available": source_available,
        "source_digest_verified": source_available and "source_digest" not in failures,
    }


class CompiledDatasetSplit:
    """Random-access view over one compiled split using memory maps."""

    def __init__(
        self,
        root: str | Path,
        split: SplitLiteral,
        *,
        verify: bool = True,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = (
            self.root if self.root.name == "manifest.json" else self.root / "manifest.json"
        )
        self.root = self.manifest_path.parent
        self.manifest = CompiledDatasetManifest.load(self.manifest_path)
        if split not in self.manifest.splits:
            raise ValueError(f"compiled dataset has no split {split!r}")
        self.split = split
        self.summary = self.manifest.splits[split]
        if verify:
            result = verify_compiled_dataset(self.root)
            if not result["valid"]:
                raise ValueError(f"compiled dataset verification failed: {result['failures']}")
        self._tokens_file = (self.root / self.summary.tokens_file).open("rb")
        self._labels_file = (self.root / self.summary.labels_file).open("rb")
        self._index_file = (self.root / self.summary.index_file).open("rb")
        self._tokens = (
            mmap.mmap(self._tokens_file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.summary.input_tokens
            else None
        )
        self._labels = (
            mmap.mmap(self._labels_file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.summary.input_tokens
            else None
        )
        self._index = (
            mmap.mmap(self._index_file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.summary.examples
            else None
        )

    def __len__(self) -> int:
        return self.summary.examples

    def __getitem__(self, index: int) -> EncodedChatExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._index is None or self._tokens is None or self._labels is None:
            raise IndexError(index)
        offset, length, target_tokens, _source_line = _ENTRY.unpack_from(
            self._index, index * _ENTRY.size
        )
        byte_offset = offset * _INT32.size
        input_ids = torch.tensor(
            struct.unpack_from(f"<{length}i", self._tokens, byte_offset), dtype=torch.long
        )
        labels = torch.tensor(
            struct.unpack_from(f"<{length}i", self._labels, byte_offset), dtype=torch.long
        )
        return EncodedChatExample(
            input_ids=input_ids,
            labels=labels,
            target_token_count=int(target_tokens),
            truncated=False,
            example_id=None,
        )

    def close(self) -> None:
        for mapping in (self._tokens, self._labels, self._index):
            if mapping is not None:
                mapping.close()
        for handle in (self._tokens_file, self._labels_file, self._index_file):
            handle.close()

    def __enter__(self) -> CompiledDatasetSplit:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(slots=True)
class CompiledStreamState:
    epoch: int = 0
    cursor: int = 0
    batches_emitted: int = 0
    examples_emitted: int = 0

    def validate(self) -> None:
        if min(self.epoch, self.cursor, self.batches_emitted, self.examples_emitted) < 0:
            raise ValueError("compiled stream counters cannot be negative")


class StatefulCompiledBatchStream(Iterator[Mapping[str, torch.Tensor]]):
    """Exact-resume compiled-data stream with an O(1)-state affine permutation."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        split: SplitLiteral = "train",
        batch_size: int = 8,
        repeat: bool = True,
        shuffle: bool = True,
        seed: int = 1337,
        pack_sequences: bool = False,
        drop_last: bool = False,
        verify: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = CompiledDatasetSplit(root, split, verify=verify)
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed
        self.pack_sequences = pack_sequences
        self.drop_last = drop_last
        self.state = CompiledStreamState()

    def __iter__(self) -> StatefulCompiledBatchStream:
        return self

    def _permutation(self) -> tuple[int, int]:
        size = len(self.dataset)
        if size <= 1 or not self.shuffle:
            return 1, 0
        rng = random.Random(canonical_hash({"seed": self.seed, "epoch": self.state.epoch}))
        candidate = rng.randrange(1, size)
        while math.gcd(candidate, size) != 1:
            candidate = (candidate + 1) % size or 1
        return candidate, rng.randrange(size)

    def _next_example(self) -> EncodedChatExample:
        size = len(self.dataset)
        if size == 0:
            raise StopIteration
        if self.state.cursor >= size:
            if not self.repeat:
                raise StopIteration
            self.state.epoch += 1
            self.state.cursor = 0
        multiplier, offset = self._permutation()
        dataset_index = (multiplier * self.state.cursor + offset) % size
        self.state.cursor += 1
        self.state.examples_emitted += 1
        return self.dataset[dataset_index]

    def __next__(self) -> Mapping[str, torch.Tensor]:
        examples: list[EncodedChatExample] = []
        while len(examples) < self.batch_size:
            try:
                examples.append(self._next_example())
            except StopIteration:
                break
        if not examples or (self.drop_last and len(examples) < self.batch_size):
            raise StopIteration
        self.state.batches_emitted += 1
        if self.pack_sequences:
            return pack_chat_examples(
                examples, max_length=self.dataset.manifest.max_length
            ).as_mapping()
        input_ids, labels, attention_mask = pad_chat_batch(examples)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def state_dict(self) -> dict[str, Any]:
        self.state.validate()
        return {
            "format_version": self.FORMAT_VERSION,
            "manifest_hash": self.dataset.manifest.to_dict()["manifest_hash"],
            "split": self.dataset.split,
            "batch_size": self.batch_size,
            "repeat": self.repeat,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "pack_sequences": self.pack_sequences,
            "drop_last": self.drop_last,
            "state": asdict(self.state),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported compiled stream-state format")
        expected = {
            "manifest_hash": self.dataset.manifest.to_dict()["manifest_hash"],
            "split": self.dataset.split,
            "batch_size": self.batch_size,
            "repeat": self.repeat,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "pack_sequences": self.pack_sequences,
            "drop_last": self.drop_last,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"compiled stream-state {name} does not match")
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("compiled stream state is missing counters")
        self.state = CompiledStreamState(**{key: int(value) for key, value in state.items()})
        self.state.validate()
        if self.state.cursor > len(self.dataset):
            raise ValueError("compiled stream cursor exceeds split size")

    def close(self) -> None:
        self.dataset.close()


__all__ = [
    "COMPILED_DATASET_FORMAT_VERSION",
    "CompiledDatasetManifest",
    "CompiledDatasetSplit",
    "CompiledSplitSummary",
    "CompiledStreamState",
    "StatefulCompiledBatchStream",
    "compile_conversation_dataset",
    "verify_compiled_dataset",
]
