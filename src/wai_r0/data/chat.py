from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from wai_r0.core.reproducibility import canonical_hash

IGNORE_INDEX = -100


class Tokenizer(Protocol):
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
class ChatExample:
    system: str
    user: str
    assistant: str
    example_id: str | None = None

    def validate(self) -> None:
        if not self.user.strip():
            raise ValueError("chat example user field cannot be empty")
        if not self.assistant.strip():
            raise ValueError("chat example assistant field cannot be empty")


@dataclass(frozen=True, slots=True)
class EncodedChatExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    target_token_count: int
    truncated: bool
    example_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids.tolist(),
            "labels": self.labels.tolist(),
            "target_token_count": self.target_token_count,
            "truncated": self.truncated,
            "example_id": self.example_id,
        }


class ByteChatTokenizer:
    """Deterministic UTF-8 byte tokenizer with explicit chat-role tokens."""

    bos_token_id = 256
    eos_token_id = 257
    system_token_id = 258
    user_token_id = 259
    assistant_token_id = 260
    vocab_size = 261

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Iterable[int]) -> str:
        byte_values = [token for token in token_ids if 0 <= token <= 255]
        return bytes(byte_values).decode("utf-8", errors="replace")

    def manifest(self) -> dict[str, Any]:
        payload = {
            "type": "byte_chat",
            "version": 1,
            "vocab_size": self.vocab_size,
            "normalization": "none; UTF-8 bytes",
            "special_tokens": {
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
                "system": self.system_token_id,
                "user": self.user_token_id,
                "assistant": self.assistant_token_id,
            },
        }
        payload["manifest_hash"] = canonical_hash(payload)
        return payload


def _segment(role_token: int, content: str, tokenizer: Tokenizer) -> list[int]:
    return [role_token, *tokenizer.encode(content), tokenizer.eos_token_id]


def _truncate_preserving_target(
    prefix: list[int],
    assistant_prefix: list[int],
    target: list[int],
    *,
    max_length: int,
    bos_token_id: int,
) -> tuple[list[int], int, bool]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not target:
        raise ValueError("assistant target cannot be empty")
    required = len(assistant_prefix) + len(target)
    truncated = len(prefix) + required > max_length
    if required >= max_length:
        retained_target = target[: max_length - len(assistant_prefix)]
        if not retained_target:
            raise ValueError("max_length leaves no assistant target token")
        sequence = [*assistant_prefix, *retained_target]
        return sequence, len(assistant_prefix), True

    prefix_budget = max_length - required
    if len(prefix) > prefix_budget:
        if prefix_budget == 0:
            retained_prefix: list[int] = []
        elif prefix_budget == 1:
            retained_prefix = [bos_token_id]
        else:
            retained_prefix = [bos_token_id, *prefix[-(prefix_budget - 1) :]]
    else:
        retained_prefix = prefix
    first_target = len(retained_prefix) + len(assistant_prefix)
    return [*retained_prefix, *assistant_prefix, *target], first_target, truncated


def encode_chat_example(
    example: ChatExample,
    *,
    tokenizer: Tokenizer | None = None,
    max_length: int = 512,
    assistant_only_loss: bool = True,
    preserve_assistant_target: bool = True,
) -> EncodedChatExample:
    """Encode one conversation with assistant-only targets by default.

    When truncation is required, the default removes old prompt context from the
    left before removing target tokens. This avoids silently producing batches
    with no supervised assistant signal.
    """

    example.validate()
    active_tokenizer = tokenizer or ByteChatTokenizer()
    prefix = [active_tokenizer.bos_token_id]
    if example.system:
        prefix.extend(_segment(active_tokenizer.system_token_id, example.system, active_tokenizer))
    prefix.extend(_segment(active_tokenizer.user_token_id, example.user, active_tokenizer))
    assistant_prefix = [active_tokenizer.assistant_token_id]
    target = [*active_tokenizer.encode(example.assistant), active_tokenizer.eos_token_id]

    if preserve_assistant_target:
        input_ids, first_target_index, truncated = _truncate_preserving_target(
            prefix,
            assistant_prefix,
            target,
            max_length=max_length,
            bos_token_id=active_tokenizer.bos_token_id,
        )
    else:
        full_sequence = [*prefix, *assistant_prefix, *target]
        truncated = len(full_sequence) > max_length
        input_ids = full_sequence[:max_length]
        first_target_index = len(prefix) + len(assistant_prefix)

    labels = list(input_ids)
    if assistant_only_loss:
        masked_prefix_length = min(first_target_index, len(labels))
        labels[:masked_prefix_length] = [IGNORE_INDEX] * masked_prefix_length
    target_count = sum(label != IGNORE_INDEX for label in labels)
    if target_count == 0:
        raise ValueError(
            "max_length truncates all assistant target tokens; increase max_length or shorten context"
        )
    return EncodedChatExample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        target_token_count=target_count,
        truncated=truncated,
        example_id=example.example_id,
    )


def pad_chat_batch(
    examples: Sequence[EncodedChatExample],
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("examples cannot be empty")
    max_length = max(item.input_ids.numel() for item in examples)
    input_ids = torch.full((len(examples), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), max_length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.bool)
    for row, item in enumerate(examples):
        length = item.input_ids.numel()
        input_ids[row, :length] = item.input_ids
        labels[row, :length] = item.labels
        attention_mask[row, :length] = True
    return input_ids, labels, attention_mask


__all__ = [
    "IGNORE_INDEX",
    "ByteChatTokenizer",
    "ChatExample",
    "EncodedChatExample",
    "Tokenizer",
    "encode_chat_example",
    "pad_chat_batch",
]
