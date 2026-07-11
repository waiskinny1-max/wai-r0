from __future__ import annotations

from dataclasses import dataclass

import torch

from wai_r0.data.chat import IGNORE_INDEX, EncodedChatExample


@dataclass(frozen=True, slots=True)
class PackedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    target_tokens: int

    def as_mapping(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
            "segment_ids": self.segment_ids,
        }


def pack_chat_examples(
    examples: list[EncodedChatExample],
    *,
    max_length: int,
    pad_token_id: int = 0,
) -> PackedBatch:
    """Greedily pack examples with block-diagonal causal attention."""

    if not examples:
        raise ValueError("examples cannot be empty")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    rows: list[list[EncodedChatExample]] = []
    current: list[EncodedChatExample] = []
    current_length = 0
    for example in examples:
        length = int(example.input_ids.numel())
        if length > max_length:
            raise ValueError("encoded example exceeds max_length")
        if current and current_length + length > max_length:
            rows.append(current)
            current = []
            current_length = 0
        current.append(example)
        current_length += length
    if current:
        rows.append(current)

    batch = len(rows)
    input_ids = torch.full((batch, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, max_length), IGNORE_INDEX, dtype=torch.long)
    segment_ids = torch.full((batch, max_length), -1, dtype=torch.long)
    position_ids = torch.zeros((batch, max_length), dtype=torch.long)
    for row_index, row in enumerate(rows):
        cursor = 0
        for segment_index, example in enumerate(row):
            length = int(example.input_ids.numel())
            stop = cursor + length
            input_ids[row_index, cursor:stop] = example.input_ids
            labels[row_index, cursor:stop] = example.labels
            # The token before a packed segment belongs to another example. Mask
            # the segment's first label so causal shifting never creates a
            # cross-example target, including full-sequence LM experiments.
            labels[row_index, cursor] = IGNORE_INDEX
            segment_ids[row_index, cursor:stop] = segment_index
            position_ids[row_index, cursor:stop] = torch.arange(length)
            cursor = stop

    valid = segment_ids.ge(0)
    same_segment = segment_ids[:, :, None].eq(segment_ids[:, None, :])
    causal = torch.ones(max_length, max_length, dtype=torch.bool).tril()
    attention_mask = same_segment & causal[None] & valid[:, :, None] & valid[:, None, :]
    return PackedBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        position_ids=position_ids,
        segment_ids=segment_ids,
        target_tokens=int(labels.ne(IGNORE_INDEX).sum()),
    )
