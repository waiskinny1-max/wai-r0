from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wai_r0.tokenization.base import Tokenizer
from wai_r0.tokenization.bpe import DeterministicBPETokenizer
from wai_r0.tokenization.byte import ByteTokenizer


def load_tokenizer(path: str | Path | None) -> Tokenizer:
    if path is None:
        return ByteTokenizer()
    source = Path(path)
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenizer artifact root must be an object")
    tokenizer_type = payload.get("tokenizer_type") or payload.get("type")
    if tokenizer_type in {"byte_chat", "byte"}:
        normalization = str(payload.get("normalization", "none"))
        if normalization not in {"none", "nfkc"}:
            raise ValueError("unsupported byte-tokenizer normalization")
        return ByteTokenizer(normalization=normalization)  # type: ignore[arg-type]
    if tokenizer_type == "deterministic_byte_bpe":
        return DeterministicBPETokenizer.from_mapping(payload)
    raise ValueError(f"unsupported tokenizer artifact type: {tokenizer_type!r}")


__all__ = ["load_tokenizer"]
