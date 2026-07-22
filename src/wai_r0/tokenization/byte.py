from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wai_r0.core.reproducibility import canonical_hash
from wai_r0.tokenization.normalization import NormalizationMode, normalize_text


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer retained as the scientific control."""

    bos_token_id = 256
    eos_token_id = 257
    system_token_id = 258
    user_token_id = 259
    assistant_token_id = 260
    vocab_size = 261

    def __init__(self, *, normalization: NormalizationMode = "none") -> None:
        self.normalization = normalization

    def encode(self, text: str) -> list[int]:
        return list(normalize_text(text, self.normalization).encode("utf-8"))

    def decode(self, token_ids: Iterable[int]) -> str:
        values = [token for token in token_ids if 0 <= int(token) <= 255]
        return bytes(values).decode("utf-8", errors="replace")

    def manifest(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "tokenizer_type": "byte_chat",
            "version": 2,
            "vocabulary_size": self.vocab_size,
            "normalization": self.normalization,
            "chat_template_version": 1,
            "special_tokens": {
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
                "system": self.system_token_id,
                "user": self.user_token_id,
                "assistant": self.assistant_token_id,
            },
            "payload": {"encoding": "utf-8", "byte_fallback": True},
            "training_corpus_hash": None,
        }
        body["manifest_hash"] = canonical_hash(body)
        return body


__all__ = ["ByteTokenizer"]
