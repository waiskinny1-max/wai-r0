from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from wai_r0.core.reproducibility import file_sha256
from wai_r0.data.chat import (
    ByteChatTokenizer,
    EncodedChatExample,
    Tokenizer,
    encode_chat_example,
    pad_chat_batch,
)
from wai_r0.data.csv_reader import iter_conversation_rows
from wai_r0.data.packing import pack_chat_examples
from wai_r0.data.splits import SplitName, SplitSpec, assign_split

STREAM_STATE_FORMAT_VERSION = 3


class _EpochBoundary(RuntimeError):
    """Internal signal used to drain one shuffle epoch before starting the next."""


@dataclass(slots=True)
class StreamState:
    epoch: int = 0
    row_cursor: int = 0
    batches_emitted: int = 0
    examples_emitted: int = 0
    matches_in_epoch: int = 0

    def validate(self) -> None:
        if (
            min(
                self.epoch,
                self.row_cursor,
                self.batches_emitted,
                self.examples_emitted,
                self.matches_in_epoch,
            )
            < 0
        ):
            raise ValueError("stream state counters cannot be negative")


class StatefulCSVBatchStream(Iterator[Mapping[str, torch.Tensor]]):
    """Bounded-memory deterministic CSV stream with exact resume semantics.

    A shuffle buffer is optional. When enabled, the buffer contents, Python RNG
    state, file cursor, and epoch counters are all checkpointed. Restoring the
    stream therefore reproduces the exact next batch rather than only returning
    to approximately the same place in the source file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: SplitName = "train",
        split_spec: SplitSpec | None = None,
        tokenizer: Tokenizer | None = None,
        batch_size: int = 8,
        max_length: int = 128,
        max_rows: int | None = None,
        repeat: bool = True,
        drop_last: bool = False,
        assistant_only_loss: bool = True,
        shuffle_buffer_size: int = 0,
        shuffle_seed: int | None = None,
        pack_sequences: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if batch_size < 1 or max_length < 2:
            raise ValueError("batch_size must be positive and max_length at least 2")
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be positive when set")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size cannot be negative")
        self.split = split
        self.split_spec = split_spec or SplitSpec()
        self.split_spec.validate()
        self.tokenizer = tokenizer or ByteChatTokenizer()
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_rows = max_rows
        self.repeat = repeat
        self.drop_last = drop_last
        self.assistant_only_loss = assistant_only_loss
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = self.split_spec.seed if shuffle_seed is None else shuffle_seed
        self.pack_sequences = pack_sequences
        self.source_sha256 = file_sha256(self.path)
        self.state = StreamState()
        self._iterator: Iterator[Any] | None = None
        self._shuffle_buffer: list[EncodedChatExample] = []
        self._random = random.Random(self.shuffle_seed)
        self._exhausted = False
        self._epoch_boundary_pending = False

    def __iter__(self) -> StatefulCSVBatchStream:
        return self

    def _open_at_cursor(self) -> Iterator[Any]:
        iterator = iter_conversation_rows(self.path, max_rows=self.max_rows)
        for _ in range(self.state.row_cursor):
            try:
                next(iterator)
            except StopIteration:
                break
        return iterator

    def _reset_epoch(self) -> None:
        self.state.epoch += 1
        self.state.row_cursor = 0
        self.state.matches_in_epoch = 0
        self._iterator = None

    def _next_matching_example(self, *, stop_at_epoch_end: bool = False) -> EncodedChatExample:
        while True:
            if self._exhausted:
                raise StopIteration
            if self._iterator is None:
                self._iterator = self._open_at_cursor()
            try:
                row = next(self._iterator)
            except StopIteration:
                if not self.repeat:
                    self._exhausted = True
                    raise
                if self.state.row_cursor == 0:
                    raise RuntimeError("CSV contains no readable rows") from None
                if self.state.matches_in_epoch == 0:
                    raise RuntimeError(f"CSV contains no rows for split {self.split!r}") from None
                self._reset_epoch()
                if stop_at_epoch_end:
                    raise _EpochBoundary from None
                continue

            self.state.row_cursor += 1
            if assign_split(row, self.split_spec) != self.split:
                continue
            self.state.matches_in_epoch += 1
            return encode_chat_example(
                row.chat,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                assistant_only_loss=self.assistant_only_loss,
            )

    def _fill_shuffle_buffer(self) -> None:
        if self._epoch_boundary_pending:
            if self._shuffle_buffer:
                return
            self._epoch_boundary_pending = False
        target_size = max(1, self.shuffle_buffer_size)
        while len(self._shuffle_buffer) < target_size:
            try:
                self._shuffle_buffer.append(self._next_matching_example(stop_at_epoch_end=True))
            except _EpochBoundary:
                self._epoch_boundary_pending = True
                break
            except StopIteration:
                break

    def _draw_example(self) -> EncodedChatExample:
        if self.shuffle_buffer_size <= 1:
            return self._next_matching_example()
        self._fill_shuffle_buffer()
        if not self._shuffle_buffer:
            raise StopIteration
        index = self._random.randrange(len(self._shuffle_buffer))
        selected = self._shuffle_buffer.pop(index)
        if not self._epoch_boundary_pending:
            try:
                self._shuffle_buffer.append(self._next_matching_example(stop_at_epoch_end=True))
            except _EpochBoundary:
                self._epoch_boundary_pending = True
            except StopIteration:
                pass
        return selected

    def __next__(self) -> Mapping[str, torch.Tensor]:
        encoded: list[EncodedChatExample] = []
        while len(encoded) < self.batch_size:
            try:
                encoded.append(self._draw_example())
            except StopIteration:
                break

        if not encoded or (self.drop_last and len(encoded) < self.batch_size):
            raise StopIteration
        self.state.batches_emitted += 1
        self.state.examples_emitted += len(encoded)
        if self.pack_sequences:
            return pack_chat_examples(encoded, max_length=self.max_length).as_mapping()
        input_ids, labels, attention_mask = pad_chat_batch(encoded)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _encoded_to_state(example: EncodedChatExample) -> dict[str, Any]:
        return example.to_dict()

    @staticmethod
    def _encoded_from_state(payload: Mapping[str, Any]) -> EncodedChatExample:
        try:
            input_ids = torch.tensor(payload.get("input_ids"), dtype=torch.long)
            labels = torch.tensor(payload.get("labels"), dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise ValueError("serialized shuffle-buffer example is malformed") from exc
        if input_ids.ndim != 1 or labels.shape != input_ids.shape or input_ids.numel() == 0:
            raise ValueError("serialized shuffle-buffer example is malformed")
        target_token_count = int(payload.get("target_token_count", -1))
        if target_token_count != int(labels.ne(-100).sum()):
            raise ValueError("serialized shuffle-buffer target count is inconsistent")
        example_id = payload.get("example_id")
        if example_id is not None and not isinstance(example_id, str):
            raise ValueError("serialized shuffle-buffer example_id must be a string or null")
        return EncodedChatExample(
            input_ids=input_ids,
            labels=labels,
            target_token_count=target_token_count,
            truncated=bool(payload.get("truncated", False)),
            example_id=example_id,
        )

    def state_dict(self) -> dict[str, Any]:
        self.state.validate()
        tokenizer_manifest = self.tokenizer.manifest()
        return {
            "format_version": STREAM_STATE_FORMAT_VERSION,
            "source": str(self.path),
            "source_sha256": self.source_sha256,
            "split": self.split,
            "split_spec": self.split_spec.to_dict(),
            "tokenizer_manifest_hash": tokenizer_manifest.get("manifest_hash"),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "max_rows": self.max_rows,
            "repeat": self.repeat,
            "drop_last": self.drop_last,
            "assistant_only_loss": self.assistant_only_loss,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "shuffle_seed": self.shuffle_seed,
            "pack_sequences": self.pack_sequences,
            "shuffle_buffer": [self._encoded_to_state(item) for item in self._shuffle_buffer],
            "shuffle_rng_state": self._random.getstate(),
            "exhausted": self._exhausted,
            "epoch_boundary_pending": self._epoch_boundary_pending,
            "state": asdict(self.state),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        format_version = payload.get("format_version")
        if format_version not in {1, 2, STREAM_STATE_FORMAT_VERSION}:
            raise ValueError("unsupported stream-state format")
        if payload.get("source_sha256") != self.source_sha256:
            raise ValueError("CSV source hash changed since the stream state was saved")
        expected: dict[str, Any] = {
            "split": self.split,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "max_rows": self.max_rows,
        }
        if int(format_version) >= 2:
            expected.update(
                {
                    "split_spec": self.split_spec.to_dict(),
                    "tokenizer_manifest_hash": self.tokenizer.manifest().get("manifest_hash"),
                    "repeat": self.repeat,
                    "drop_last": self.drop_last,
                    "assistant_only_loss": self.assistant_only_loss,
                }
            )
        if int(format_version) >= 3:
            expected.update(
                {
                    "shuffle_buffer_size": self.shuffle_buffer_size,
                    "shuffle_seed": self.shuffle_seed,
                    "pack_sequences": self.pack_sequences,
                }
            )
        elif self.shuffle_buffer_size > 1 or self.pack_sequences:
            raise ValueError("legacy stream state cannot restore shuffle or packing configuration")
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"stream-state {name} does not match current configuration")

        state_payload = payload.get("state")
        if not isinstance(state_payload, Mapping):
            raise ValueError("stream state payload is missing")
        unknown = sorted(set(state_payload) - set(StreamState.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown stream-state counters: {', '.join(unknown)}")
        state_defaults = asdict(StreamState())
        state_defaults.update(dict(state_payload))
        self.state = StreamState(**state_defaults)
        self.state.validate()

        if int(format_version) >= 3:
            buffer_payload = payload.get("shuffle_buffer")
            rng_state = payload.get("shuffle_rng_state")
            if not isinstance(buffer_payload, list) or rng_state is None:
                raise ValueError("stream state is missing shuffle-buffer state")
            self._shuffle_buffer = [
                self._encoded_from_state(item)
                for item in buffer_payload
                if isinstance(item, Mapping)
            ]
            if len(self._shuffle_buffer) != len(buffer_payload):
                raise ValueError("stream shuffle buffer contains a non-mapping entry")
            if len(self._shuffle_buffer) > self.shuffle_buffer_size:
                raise ValueError("stream shuffle buffer exceeds configured capacity")
            self._random.setstate(_nested_tuple(rng_state))
            self._exhausted = bool(payload.get("exhausted", False))
            self._epoch_boundary_pending = bool(payload.get("epoch_boundary_pending", False))
            if self._epoch_boundary_pending and not self.repeat:
                raise ValueError("non-repeating stream cannot have a pending epoch boundary")
        else:
            self._shuffle_buffer = []
            self._random = random.Random(self.shuffle_seed)
            self._exhausted = False
            self._epoch_boundary_pending = False
        self._iterator = None


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


__all__ = ["STREAM_STATE_FORMAT_VERSION", "StatefulCSVBatchStream", "StreamState"]
