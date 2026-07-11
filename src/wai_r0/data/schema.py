from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from wai_r0.core.reproducibility import canonical_hash
from wai_r0.data.chat import ChatExample

CANONICAL_FIELDS = (
    "id",
    "split",
    "task_family",
    "difficulty",
    "system",
    "user",
    "assistant",
    "answer_format",
    "eval_type",
    "metadata_json",
)
REQUIRED_FIELDS = ("user", "assistant")
SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "training": "train",
    "testing": "test",
}


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())


def normalize_split(value: str) -> str:
    normalized = value.strip().lower()
    return SPLIT_ALIASES.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class ConversationRow:
    id: str
    split: str
    task_family: str
    difficulty: str
    system: str
    user: str
    assistant: str
    answer_format: str
    eval_type: str
    metadata: dict[str, Any]
    source_line: int | None = None

    def validate(self, *, max_field_chars: int = 1_000_000) -> None:
        if not self.user.strip():
            raise ValueError("user field is empty")
        if not self.assistant.strip():
            raise ValueError("assistant field is empty")
        if max_field_chars < 1:
            raise ValueError("max_field_chars must be positive")
        for name in ("system", "user", "assistant"):
            if len(getattr(self, name)) > max_field_chars:
                raise ValueError(f"{name} exceeds max_field_chars")
        try:
            json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be finite JSON-compatible data") from exc

    @property
    def normalized_split(self) -> str:
        return normalize_split(self.split)

    @property
    def chat(self) -> ChatExample:
        return ChatExample(
            system=self.system,
            user=self.user,
            assistant=self.assistant,
            example_id=self.id or None,
        )

    @property
    def normalized_content(self) -> str:
        return "\n".join(
            (
                normalize_text(self.system),
                normalize_text(self.user),
                normalize_text(self.assistant),
            )
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "system": normalize_text(self.system),
                "user": normalize_text(self.user),
                "assistant": normalize_text(self.assistant),
            }
        )

    @property
    def group_key(self) -> str:
        for key in ("group_id", "conversation_id", "source_id", "parent_id"):
            value = self.metadata.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return f"metadata:{key}:{value}"
        if self.id:
            return f"id:{self.id}"
        return f"content:{self.content_hash}"
