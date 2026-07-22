from __future__ import annotations

import unicodedata
from typing import Literal

NormalizationMode = Literal["none", "nfkc"]


def normalize_text(text: str, mode: NormalizationMode = "none") -> str:
    if mode == "none":
        return text
    if mode == "nfkc":
        return unicodedata.normalize("NFKC", text)
    raise ValueError(f"unsupported normalization mode: {mode}")


__all__ = ["NormalizationMode", "normalize_text"]
