from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from wai_r0.core.reproducibility import atomic_write_json, canonical_hash

DATA_MANIFEST_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class LengthSummary:
    count: int
    minimum: int
    maximum: int
    mean: float
    p50: float
    p90: float
    p99: float


@dataclass(slots=True)
class DatasetManifest:
    source: str
    source_sha256: str
    source_bytes: int
    schema_fields: list[str]
    accepted_rows: int
    rejected_rows: int
    split_counts: dict[str, int]
    task_family_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    exact_duplicate_content_rows: int
    near_duplicate_content_rows: int
    length_summaries: dict[str, LengthSummary]
    split_policy: dict[str, Any]
    limitations: list[str] = field(default_factory=list)
    manifest_version: str = DATA_MANIFEST_VERSION

    def validate(self) -> None:
        if self.manifest_version != DATA_MANIFEST_VERSION:
            raise ValueError(f"unsupported dataset manifest version: {self.manifest_version}")
        if self.accepted_rows < 0 or self.rejected_rows < 0:
            raise ValueError("dataset row counts cannot be negative")
        if not self.source_sha256 or len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        if sum(self.split_counts.values()) != self.accepted_rows:
            raise ValueError("split counts must sum to accepted_rows")
        if not self.limitations:
            raise ValueError("dataset manifests must state at least one limitation")

    @property
    def manifest_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        return canonical_hash(payload)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        if include_hash:
            payload["manifest_hash"] = canonical_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        normalized = dict(payload)
        expected_hash = normalized.pop("manifest_hash", None)
        lengths_payload = normalized.get("length_summaries", {})
        if not isinstance(lengths_payload, dict):
            raise ValueError("length_summaries must be a mapping")
        normalized["length_summaries"] = {
            name: value if isinstance(value, LengthSummary) else LengthSummary(**value)
            for name, value in lengths_payload.items()
        }
        manifest = cls(**normalized)
        manifest.validate()
        if expected_hash is not None and expected_hash != manifest.manifest_hash:
            raise ValueError("dataset manifest hash does not match its contents")
        return manifest


def write_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> Path:
    return atomic_write_json(path, manifest.to_dict())
