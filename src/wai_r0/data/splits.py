from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal, cast

from wai_r0.data.schema import ConversationRow

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SplitSpec:
    train: float = 0.90
    val: float = 0.05
    test: float = 0.05
    seed: int = 1337
    respect_declared: bool = False
    group_aware: bool = True

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split fractions cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        if self.train <= 0:
            raise ValueError("training split must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def _unit_interval(key: str, seed: int) -> float:
    digest = hashlib.blake2b(
        key.encode("utf-8"), digest_size=8, person=b"wai-r0-split", key=str(seed).encode()
    ).digest()
    integer = int.from_bytes(digest, "big", signed=False)
    return integer / float(2**64)


def assign_split(row: ConversationRow, spec: SplitSpec) -> SplitName:
    spec.validate()
    declared = row.normalized_split
    if spec.respect_declared and declared in {"train", "val", "test"}:
        return cast(SplitName, declared)
    key = row.group_key if spec.group_aware else row.content_hash
    value = _unit_interval(key, spec.seed)
    if value < spec.train:
        return "train"
    if value < spec.train + spec.val:
        return "val"
    return "test"
