from __future__ import annotations

import csv

import pytest
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.data.chat import ChatExample, encode_chat_example
from wai_r0.data.packing import pack_chat_examples
from wai_r0.data.splits import SplitSpec
from wai_r0.data.streaming import StatefulCSVBatchStream
from wai_r0.model import ModelOutput, ReasonerCore


def _write_chat_csv(path, count: int = 20) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "system", "user", "assistant", "metadata_json"],
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "id": str(index),
                    "system": "be useful",
                    "user": f"question {index}",
                    "assistant": f"answer {index}",
                    "metadata_json": "{}",
                }
            )


def test_csv_stream_exact_resume(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path)
    spec = SplitSpec(train=1.0, val=0.0, test=0.0, seed=1)
    stream = StatefulCSVBatchStream(path, split_spec=spec, batch_size=3, max_length=64)
    next(stream)
    state = stream.state_dict()
    expected = next(stream)

    restored = StatefulCSVBatchStream(path, split_spec=spec, batch_size=3, max_length=64)
    restored.load_state_dict(state)
    actual = next(restored)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_csv_stream_missing_split_fails_in_one_pass(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=3)
    stream = StatefulCSVBatchStream(
        path,
        split="val",
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0),
        batch_size=1,
        max_length=64,
    )
    with pytest.raises(RuntimeError, match="no rows for split"):
        next(stream)


def test_packed_segments_are_attention_isolated() -> None:
    config = ReasonerConfig(
        vocab_size=261,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=32,
        max_seq_len=64,
        dropout=0.0,
        seed=9,
    )
    model = ReasonerCore(config).transformer.eval()
    first = encode_chat_example(ChatExample("", "alpha", "one"), max_length=32)
    second = encode_chat_example(ChatExample("", "beta", "two"), max_length=32)
    packed = pack_chat_examples([first, second], max_length=64)
    changed = packed.input_ids.clone()
    first_segment = packed.segment_ids.eq(0)
    changed[first_segment] = (changed[first_segment] + 7) % config.vocab_size

    with torch.inference_mode():
        original = model(
            packed.input_ids,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            return_dict=True,
        )
        altered = model(
            changed,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            return_dict=True,
        )
    assert isinstance(original, ModelOutput) and isinstance(altered, ModelOutput)
    second_segment = packed.segment_ids.eq(1)
    torch.testing.assert_close(
        original.logits[second_segment], altered.logits[second_segment], rtol=0, atol=0
    )


def test_csv_stream_state_rejects_semantic_configuration_changes(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=8)
    original = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0, seed=7),
        batch_size=2,
        max_length=48,
        assistant_only_loss=True,
    )
    next(original)
    state = original.state_dict()
    assert state["format_version"] == 3

    changed_split = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0, seed=8),
        batch_size=2,
        max_length=48,
    )
    with pytest.raises(ValueError, match="split_spec"):
        changed_split.load_state_dict(state)

    changed_objective = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0, seed=7),
        batch_size=2,
        max_length=48,
        assistant_only_loss=False,
    )
    with pytest.raises(ValueError, match="assistant_only_loss"):
        changed_objective.load_state_dict(state)

    malformed = dict(state)
    malformed["state"] = {**state["state"], "unexpected": 1}
    with pytest.raises(ValueError, match="unknown stream-state"):
        original.load_state_dict(malformed)


def test_full_sequence_packing_masks_every_segment_boundary() -> None:
    first = encode_chat_example(
        ChatExample("", "alpha", "one"),
        max_length=32,
        assistant_only_loss=False,
    )
    second = encode_chat_example(
        ChatExample("", "beta", "two"),
        max_length=32,
        assistant_only_loss=False,
    )
    packed = pack_chat_examples([first, second], max_length=64)
    segment_starts = []
    previous = -1
    for index, segment in enumerate(packed.segment_ids[0].tolist()):
        if segment >= 0 and segment != previous:
            segment_starts.append(index)
        previous = segment
    assert len(segment_starts) == 2
    assert all(int(packed.labels[0, index]) == -100 for index in segment_starts)


def test_shuffled_csv_stream_exact_resume(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=17)
    spec = SplitSpec(train=1.0, val=0.0, test=0.0, seed=19)
    options = {
        "split_spec": spec,
        "batch_size": 4,
        "max_length": 64,
        "shuffle_buffer_size": 7,
        "shuffle_seed": 12345,
    }
    stream = StatefulCSVBatchStream(path, **options)
    next(stream)
    state = stream.state_dict()
    expected = next(stream)

    restored = StatefulCSVBatchStream(path, **options)
    restored.load_state_dict(state)
    actual = next(restored)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_packed_csv_stream_emits_structured_masks(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=8)
    stream = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0),
        batch_size=3,
        max_length=96,
        pack_sequences=True,
    )
    batch = next(stream)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].ndim == 3
    assert batch["position_ids"].shape == batch["input_ids"].shape
    assert batch["segment_ids"].shape == batch["input_ids"].shape


def test_stream_state_rejects_shuffle_and_packing_changes(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=10)
    spec = SplitSpec(train=1.0, val=0.0, test=0.0, seed=4)
    original = StatefulCSVBatchStream(
        path,
        split_spec=spec,
        batch_size=2,
        max_length=64,
        shuffle_buffer_size=5,
        shuffle_seed=88,
        pack_sequences=True,
    )
    next(original)
    state = original.state_dict()

    changed_seed = StatefulCSVBatchStream(
        path,
        split_spec=spec,
        batch_size=2,
        max_length=64,
        shuffle_buffer_size=5,
        shuffle_seed=89,
        pack_sequences=True,
    )
    with pytest.raises(ValueError, match="shuffle_seed"):
        changed_seed.load_state_dict(state)

    changed_packing = StatefulCSVBatchStream(
        path,
        split_spec=spec,
        batch_size=2,
        max_length=64,
        shuffle_buffer_size=5,
        shuffle_seed=88,
        pack_sequences=False,
    )
    with pytest.raises(ValueError, match="pack_sequences"):
        changed_packing.load_state_dict(state)


def test_malformed_shuffle_buffer_state_fails_closed(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=6)
    stream = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0),
        batch_size=2,
        max_length=64,
        shuffle_buffer_size=3,
    )
    next(stream)
    state = stream.state_dict()
    state["shuffle_buffer"] = [{"input_ids": None, "labels": None}]
    restored = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0),
        batch_size=2,
        max_length=64,
        shuffle_buffer_size=3,
    )
    with pytest.raises(ValueError, match="malformed"):
        restored.load_state_dict(state)


def test_shuffle_buffer_does_not_mix_multiple_epochs_during_initial_fill(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=5)
    stream = StatefulCSVBatchStream(
        path,
        split_spec=SplitSpec(train=1.0, val=0.0, test=0.0),
        batch_size=1,
        max_length=64,
        shuffle_buffer_size=100,
        shuffle_seed=3,
    )
    first_epoch = [tuple(next(stream)["input_ids"][0].tolist()) for _ in range(5)]
    assert len(set(first_epoch)) == 5
    first_next_epoch = tuple(next(stream)["input_ids"][0].tolist())
    assert first_next_epoch in set(first_epoch)


def test_pending_epoch_boundary_is_exactly_resumable(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    _write_chat_csv(path, count=4)
    options = {
        "split_spec": SplitSpec(train=1.0, val=0.0, test=0.0),
        "batch_size": 1,
        "max_length": 64,
        "shuffle_buffer_size": 20,
        "shuffle_seed": 91,
    }
    stream = StatefulCSVBatchStream(path, **options)
    next(stream)
    state = stream.state_dict()
    assert state["epoch_boundary_pending"] is True
    expected = next(stream)

    restored = StatefulCSVBatchStream(path, **options)
    restored.load_state_dict(state)
    actual = next(restored)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_packed_target_count_matches_effective_labels() -> None:
    examples = [
        encode_chat_example(
            ChatExample("", "alpha", "one"),
            max_length=32,
            assistant_only_loss=False,
        ),
        encode_chat_example(
            ChatExample("", "beta", "two"),
            max_length=32,
            assistant_only_loss=False,
        ),
    ]
    packed = pack_chat_examples(examples, max_length=64)
    assert packed.target_tokens == int(packed.labels.ne(-100).sum())
    assert packed.target_tokens == sum(item.target_token_count for item in examples) - 2
