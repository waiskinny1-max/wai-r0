from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def normalized_fingerprint_text(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def exact_text_hash(text: str) -> str:
    return hashlib.sha256(normalized_fingerprint_text(text).encode("utf-8")).hexdigest()


def simhash64(text: str) -> int:
    tokens = _TOKEN.findall(text.casefold())
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    exact: bool
    near: bool
    matched_index: int | None
    distance: int | None = None


class DuplicateIndex:
    """Bounded-candidate exact and SimHash near-duplicate index.

    SimHash is partitioned into four 16-bit bands. Candidates only need one
    matching band, avoiding an O(n²) scan for large local CSV files.
    """

    def __init__(self, *, near_distance: int = 3, max_bucket_size: int = 2048) -> None:
        if not 0 <= near_distance <= 16:
            raise ValueError("near_distance must be between 0 and 16")
        if max_bucket_size < 1:
            raise ValueError("max_bucket_size must be positive")
        self.near_distance = near_distance
        self.max_bucket_size = max_bucket_size
        self._exact: dict[str, int] = {}
        self._fingerprints: list[int] = []
        self._bands: dict[tuple[int, int], list[int]] = defaultdict(list)

    def add(self, text: str) -> DuplicateMatch:
        exact_hash = exact_text_hash(text)
        if exact_hash in self._exact:
            return DuplicateMatch(True, True, self._exact[exact_hash], 0)

        fingerprint = simhash64(text)
        candidates: set[int] = set()
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            candidates.update(self._bands.get((band, value), ()))
        best_index: int | None = None
        best_distance: int | None = None
        for index in candidates:
            distance = hamming_distance(fingerprint, self._fingerprints[index])
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        near = best_distance is not None and best_distance <= self.near_distance

        index = len(self._fingerprints)
        self._exact[exact_hash] = index
        self._fingerprints.append(fingerprint)
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            bucket = self._bands[(band, value)]
            if len(bucket) < self.max_bucket_size:
                bucket.append(index)
        return DuplicateMatch(False, near, best_index if near else None, best_distance)

    def __len__(self) -> int:
        return len(self._fingerprints)
