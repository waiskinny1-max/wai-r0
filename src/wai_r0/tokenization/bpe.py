from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from wai_r0.core.reproducibility import atomic_write_json, canonical_hash
from wai_r0.tokenization.base import TokenizerArtifact
from wai_r0.tokenization.normalization import NormalizationMode, normalize_text

_BYTE_VOCABULARY = 256
_BOS = 256
_EOS = 257
_SYSTEM = 258
_USER = 259
_ASSISTANT = 260
_FIRST_MERGE = 261


@dataclass(frozen=True, slots=True)
class BPETrainingSummary:
    requested_vocab_size: int
    actual_vocab_size: int
    merge_count: int
    input_sequences: int
    input_bytes: int
    corpus_hash: str
    stopped_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_vocab_size": self.requested_vocab_size,
            "actual_vocab_size": self.actual_vocab_size,
            "merge_count": self.merge_count,
            "input_sequences": self.input_sequences,
            "input_bytes": self.input_bytes,
            "corpus_hash": self.corpus_hash,
            "stopped_reason": self.stopped_reason,
        }


class DeterministicBPETokenizer:
    """Small deterministic byte-level BPE implementation.

    The base vocabulary is every byte plus five fixed chat-role tokens. Merges
    are applied by rank. The implementation prioritizes reproducibility and
    transparent artifacts over tokenizer-training speed.
    """

    bos_token_id = _BOS
    eos_token_id = _EOS
    system_token_id = _SYSTEM
    user_token_id = _USER
    assistant_token_id = _ASSISTANT

    def __init__(
        self,
        merges: Sequence[tuple[int, int]],
        *,
        normalization: NormalizationMode = "none",
        training_corpus_hash: str | None = None,
        chat_template_version: int = 1,
    ) -> None:
        normalized_merges = tuple((int(left), int(right)) for left, right in merges)
        for rank, (left, right) in enumerate(normalized_merges):
            upper_bound = _FIRST_MERGE + rank
            if left < 0 or right < 0 or left >= upper_bound or right >= upper_bound:
                raise ValueError("BPE merge references an unavailable token")
        self.merges = normalized_merges
        self.normalization = normalization
        self.training_corpus_hash = training_corpus_hash
        self.chat_template_version = chat_template_version
        self._pair_to_token = {pair: _FIRST_MERGE + rank for rank, pair in enumerate(self.merges)}
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = _FIRST_MERGE + len(self.merges)
        self._expansions = self._build_expansions()

    def _build_expansions(self) -> dict[int, bytes]:
        expansions: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for rank, (left, right) in enumerate(self.merges):
            token = _FIRST_MERGE + rank
            expansions[token] = expansions[left] + expansions[right]
        return expansions

    def encode(self, text: str) -> list[int]:
        tokens = list(normalize_text(text, self.normalization).encode("utf-8"))
        if len(tokens) < 2 or not self.merges:
            return tokens
        return self._merge_tokens(tokens)

    def _merge_tokens(self, tokens: list[int]) -> list[int]:
        """Apply ranked merges with a linked-list occurrence heap.

        A rank is processed for every currently valid occurrence from left to
        right, preserving the reference BPE semantics while avoiding repeated
        full-sequence scans for each applied merge.
        """

        size = len(tokens)
        previous = [index - 1 for index in range(size)]
        following = [index + 1 for index in range(size)]
        following[-1] = -1
        active = [True] * size
        occurrences: list[tuple[int, int, int, int]] = []

        def push_occurrence(left_index: int) -> None:
            if left_index < 0 or not active[left_index]:
                return
            right_index = following[left_index]
            if right_index < 0 or not active[right_index]:
                return
            left_token = tokens[left_index]
            right_token = tokens[right_index]
            rank = self._merge_rank.get((left_token, right_token))
            if rank is not None:
                heapq.heappush(
                    occurrences,
                    (rank, left_index, left_token, right_token),
                )

        for index in range(size - 1):
            push_occurrence(index)

        while occurrences:
            rank = occurrences[0][0]
            ranked_occurrences: list[tuple[int, int, int, int]] = []
            while occurrences and occurrences[0][0] == rank:
                ranked_occurrences.append(heapq.heappop(occurrences))

            replacement = _FIRST_MERGE + rank
            for _rank, left_index, expected_left, expected_right in ranked_occurrences:
                if not active[left_index] or tokens[left_index] != expected_left:
                    continue
                right_index = following[left_index]
                if (
                    right_index < 0
                    or not active[right_index]
                    or tokens[right_index] != expected_right
                ):
                    continue

                left_neighbor = previous[left_index]
                right_neighbor = following[right_index]
                tokens[left_index] = replacement
                active[right_index] = False
                following[left_index] = right_neighbor
                if right_neighbor >= 0:
                    previous[right_neighbor] = left_index

                push_occurrence(left_neighbor)
                push_occurrence(left_index)

        merged: list[int] = []
        index = 0
        while index >= 0:
            if active[index]:
                merged.append(tokens[index])
            index = following[index]
        return merged

    def decode(self, token_ids: Iterable[int]) -> str:
        chunks: list[bytes] = []
        for raw_token in token_ids:
            token = int(raw_token)
            expansion = self._expansions.get(token)
            if expansion is not None:
                chunks.append(expansion)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def artifact(self) -> TokenizerArtifact:
        return TokenizerArtifact(
            tokenizer_type="deterministic_byte_bpe",
            version=1,
            vocabulary_size=self.vocab_size,
            special_tokens={
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
                "system": self.system_token_id,
                "user": self.user_token_id,
                "assistant": self.assistant_token_id,
            },
            normalization=self.normalization,
            payload={"merges": [list(pair) for pair in self.merges], "byte_fallback": True},
            training_corpus_hash=self.training_corpus_hash,
            chat_template_version=self.chat_template_version,
        )

    def manifest(self) -> dict[str, Any]:
        return self.artifact().to_dict()

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.manifest())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DeterministicBPETokenizer:
        artifact = TokenizerArtifact.from_mapping(payload)
        if artifact.tokenizer_type != "deterministic_byte_bpe":
            raise ValueError("tokenizer artifact is not deterministic_byte_bpe")
        raw_merges = artifact.payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("BPE artifact is missing merges")
        merges: list[tuple[int, int]] = []
        for item in raw_merges:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("BPE merge entry must contain two token IDs")
            merges.append((int(item[0]), int(item[1])))
        return cls(
            merges,
            normalization=artifact.normalization,  # type: ignore[arg-type]
            training_corpus_hash=artifact.training_corpus_hash,
            chat_template_version=artifact.chat_template_version,
        )

    @classmethod
    def load(cls, path: str | Path) -> DeterministicBPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("tokenizer artifact root must be an object")
        return cls.from_mapping(payload)


def _merge_pair(sequence: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            merged.append(replacement)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


def train_deterministic_bpe(
    corpus: Iterable[str],
    *,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    normalization: NormalizationMode = "none",
    max_training_bytes: int = 16_000_000,
) -> tuple[DeterministicBPETokenizer, BPETrainingSummary]:
    if vocab_size < _FIRST_MERGE:
        raise ValueError(f"vocab_size must be at least {_FIRST_MERGE}")
    if min_frequency < 2:
        raise ValueError("min_frequency must be at least 2")
    if max_training_bytes < 1:
        raise ValueError("max_training_bytes must be positive")

    sequences: list[list[int]] = []
    corpus_records: list[str] = []
    input_bytes = 0
    stopped_reason = "target_vocab_reached"
    for text in corpus:
        normalized = normalize_text(str(text), normalization)
        encoded = list(normalized.encode("utf-8"))
        if not encoded:
            continue
        if input_bytes + len(encoded) > max_training_bytes:
            stopped_reason = "training_byte_limit_reached"
            break
        sequences.append(encoded)
        corpus_records.append(normalized)
        input_bytes += len(encoded)

    if not sequences:
        raise ValueError("tokenizer training corpus contains no non-empty text")
    corpus_hash = canonical_hash(corpus_records)
    merges: list[tuple[int, int]] = []
    while _FIRST_MERGE + len(merges) < vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            pair_counts.update(pairwise(sequence))
        if not pair_counts:
            stopped_reason = "no_pairs_remaining"
            break
        max_frequency = max(pair_counts.values())
        if max_frequency < min_frequency:
            stopped_reason = "min_frequency_not_met"
            break
        candidates = [pair for pair, count in pair_counts.items() if count == max_frequency]
        selected = min(candidates)
        replacement = _FIRST_MERGE + len(merges)
        sequences = [_merge_pair(sequence, selected, replacement) for sequence in sequences]
        merges.append(selected)

    tokenizer = DeterministicBPETokenizer(
        merges,
        normalization=normalization,
        training_corpus_hash=corpus_hash,
    )
    summary = BPETrainingSummary(
        requested_vocab_size=vocab_size,
        actual_vocab_size=tokenizer.vocab_size,
        merge_count=len(merges),
        input_sequences=len(sequences),
        input_bytes=input_bytes,
        corpus_hash=corpus_hash,
        stopped_reason=stopped_reason,
    )
    return tokenizer, summary


__all__ = [
    "BPETrainingSummary",
    "DeterministicBPETokenizer",
    "train_deterministic_bpe",
]
