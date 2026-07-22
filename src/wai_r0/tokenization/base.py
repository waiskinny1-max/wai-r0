from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wai_r0.core.reproducibility import canonical_hash


@runtime_checkable
class Tokenizer(Protocol):
    """Stable tokenizer contract used by compiled datasets and inference."""

    bos_token_id: int
    eos_token_id: int
    system_token_id: int
    user_token_id: int
    assistant_token_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...

    def manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TokenizerArtifact:
    """Serializable tokenizer plus immutable provenance metadata."""

    tokenizer_type: str
    version: int
    vocabulary_size: int
    special_tokens: dict[str, int]
    normalization: str
    payload: dict[str, Any]
    training_corpus_hash: str | None = None
    chat_template_version: int = 1

    def validate(self) -> None:
        if not self.tokenizer_type.strip():
            raise ValueError("tokenizer_type cannot be empty")
        if self.version < 1:
            raise ValueError("tokenizer version must be positive")
        if self.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        required = {"bos", "eos", "system", "user", "assistant"}
        if set(self.special_tokens) != required:
            raise ValueError("special_tokens must define bos/eos/system/user/assistant exactly")
        ids = list(self.special_tokens.values())
        if len(set(ids)) != len(ids) or any(token < 0 for token in ids):
            raise ValueError("special token IDs must be unique non-negative integers")
        if any(token >= self.vocabulary_size for token in ids):
            raise ValueError("special token IDs must be inside the vocabulary")
        if self.chat_template_version < 1:
            raise ValueError("chat_template_version must be positive")
        canonical_hash(self.payload)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        body = {
            "tokenizer_type": self.tokenizer_type,
            "version": self.version,
            "vocabulary_size": self.vocabulary_size,
            "special_tokens": dict(sorted(self.special_tokens.items())),
            "normalization": self.normalization,
            "payload": self.payload,
            "training_corpus_hash": self.training_corpus_hash,
            "chat_template_version": self.chat_template_version,
        }
        body["manifest_hash"] = canonical_hash(body)
        return body

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TokenizerArtifact:
        manifest_hash = payload.get("manifest_hash")
        body = {key: value for key, value in payload.items() if key != "manifest_hash"}
        artifact = cls(
            tokenizer_type=str(body.get("tokenizer_type", "")),
            version=int(body.get("version", 0)),
            vocabulary_size=int(body.get("vocabulary_size", 0)),
            special_tokens={
                str(key): int(value) for key, value in dict(body.get("special_tokens", {})).items()
            },
            normalization=str(body.get("normalization", "")),
            payload=dict(body.get("payload", {})),
            training_corpus_hash=(
                None
                if body.get("training_corpus_hash") is None
                else str(body.get("training_corpus_hash"))
            ),
            chat_template_version=int(body.get("chat_template_version", 1)),
        )
        artifact.validate()
        if manifest_hash is not None and str(manifest_hash) != canonical_hash(body):
            raise ValueError("tokenizer manifest hash does not match its payload")
        return artifact


__all__ = ["Tokenizer", "TokenizerArtifact"]
