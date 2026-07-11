from __future__ import annotations

import csv
import json

import pytest
import torch

from wai_r0.data.chat import (
    IGNORE_INDEX,
    ByteChatTokenizer,
    ChatExample,
    encode_chat_example,
    pad_chat_batch,
)
from wai_r0.data.csv_reader import audit_conversation_csv, iter_conversation_rows
from wai_r0.training.losses import causal_language_model_loss


def test_assistant_only_mask_excludes_prompt() -> None:
    tokenizer = ByteChatTokenizer()
    encoded = encode_chat_example(
        ChatExample(system="rule", user="question", assistant="answer"),
        tokenizer=tokenizer,
        max_length=128,
    )
    target_positions = encoded.labels.ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
    assert target_positions.numel() == len(tokenizer.encode("answer")) + 1
    first_target = int(target_positions[0])
    assert encoded.input_ids[first_target - 1] == tokenizer.assistant_token_id
    assert torch.equal(encoded.input_ids[target_positions], encoded.labels[target_positions])


def test_truncation_preserves_at_least_one_target_token() -> None:
    encoded = encode_chat_example(
        ChatExample(system="long" * 20, user="question", assistant="answer"),
        max_length=4,
    )
    assert encoded.truncated is True
    assert encoded.target_token_count > 0
    assert encoded.input_ids.numel() == 4


def test_padding_masks_labels() -> None:
    examples = [
        encode_chat_example(ChatExample("", "a", "b"), max_length=32),
        encode_chat_example(ChatExample("", "longer", "reply"), max_length=32),
    ]
    input_ids, labels, attention_mask = pad_chat_batch(examples)
    assert input_ids.shape == labels.shape == attention_mask.shape
    assert labels[~attention_mask].eq(IGNORE_INDEX).all()


def test_causal_loss_uses_supervised_targets() -> None:
    encoded = encode_chat_example(ChatExample("", "a", "b"), max_length=32)
    input_ids, labels, _ = pad_chat_batch([encoded])
    logits = torch.randn(1, input_ids.shape[1], ByteChatTokenizer.vocab_size)
    loss, components = causal_language_model_loss(logits, labels)
    assert torch.isfinite(loss)
    assert set(components) == {"language_model"}


def test_csv_stream_and_audit(tmp_path) -> None:
    path = tmp_path / "data.csv"
    fieldnames = [
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
    ]
    rows = [
        {
            "id": "a",
            "split": "train",
            "task_family": "chat",
            "difficulty": "easy",
            "system": "be clear",
            "user": "hello",
            "assistant": "hi",
            "answer_format": "text",
            "eval_type": "semantic",
            "metadata_json": json.dumps({"seed": 1}),
        },
        {
            "id": "a",
            "split": "train",
            "task_family": "chat",
            "difficulty": "easy",
            "system": "",
            "user": "duplicate",
            "assistant": "row",
            "answer_format": "text",
            "eval_type": "semantic",
            "metadata_json": "{}",
        },
        {
            "id": "c",
            "split": "dev",
            "task_family": "rewrite",
            "difficulty": "hard",
            "system": "",
            "user": "",
            "assistant": "missing prompt",
            "answer_format": "text",
            "eval_type": "exact",
            "metadata_json": "not-json",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    audit = audit_conversation_csv(path)
    assert audit.total_rows == 3
    assert audit.accepted_rows == 1
    assert audit.rejected_rows == 2
    assert audit.duplicate_ids == 1
    assert audit.empty_user_rows == 1
    assert audit.invalid_metadata_rows == 1

    with pytest.raises(ValueError, match="line 4"):
        list(iter_conversation_rows(path))
