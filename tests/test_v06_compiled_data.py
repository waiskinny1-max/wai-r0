from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wai_r0.data.compiled import (
    CompiledDatasetManifest,
    CompiledDatasetSplit,
    StatefulCompiledBatchStream,
    compile_conversation_dataset,
    verify_compiled_dataset,
)
from wai_r0.data.splits import SplitSpec
from wai_r0.tokenization import ByteTokenizer


def _write_csv(path: Path) -> None:
    rows = ["id,split,system,user,assistant"]
    for index in range(20):
        split = "train" if index < 14 else ("val" if index < 17 else "test")
        rows.append(f"{index},{split},Be exact,prompt {index},answer {index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_compile_verify_and_random_access(tmp_path: Path) -> None:
    csv_path = tmp_path / "chat.csv"
    _write_csv(csv_path)
    root = tmp_path / "compiled"
    manifest_path = compile_conversation_dataset(
        csv_path,
        output_dir=root,
        tokenizer=ByteTokenizer(),
        split_spec=SplitSpec(respect_declared=True),
        max_length=64,
    )
    result = verify_compiled_dataset(root)
    assert result["valid"] is True
    manifest = CompiledDatasetManifest.load(manifest_path)
    assert manifest.splits["train"].examples == 14
    assert manifest.splits["val"].examples == 3
    assert manifest.splits["test"].examples == 3
    assert manifest.splits["train"].raw_utf8_bytes > 0
    assert manifest.splits["train"].target_utf8_bytes > 0
    assert manifest.splits["train"].target_utf8_bytes < manifest.splits["train"].raw_utf8_bytes

    with CompiledDatasetSplit(root, "train") as dataset:
        example = dataset[0]
        assert example.input_ids.ndim == 1
        assert example.input_ids.shape == example.labels.shape
        assert example.target_token_count > 0


def test_compiled_stream_exact_resume(tmp_path: Path) -> None:
    csv_path = tmp_path / "chat.csv"
    _write_csv(csv_path)
    root = tmp_path / "compiled"
    compile_conversation_dataset(
        csv_path,
        output_dir=root,
        tokenizer=ByteTokenizer(),
        split_spec=SplitSpec(respect_declared=True),
        max_length=64,
    )
    first = StatefulCompiledBatchStream(root, batch_size=3, seed=99, pack_sequences=False)
    _ = next(first)
    state = first.state_dict()
    expected = next(first)

    restored = StatefulCompiledBatchStream(root, batch_size=3, seed=99, pack_sequences=False)
    restored.load_state_dict(state)
    actual = next(restored)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    first.close()
    restored.close()


def test_compiled_verifier_detects_corruption(tmp_path: Path) -> None:
    csv_path = tmp_path / "chat.csv"
    _write_csv(csv_path)
    root = tmp_path / "compiled"
    compile_conversation_dataset(
        csv_path,
        output_dir=root,
        tokenizer=ByteTokenizer(),
        split_spec=SplitSpec(respect_declared=True),
        max_length=64,
    )
    tokens = next((root / "shards").glob("train-*.tokens.bin"))
    tokens.write_bytes(tokens.read_bytes() + b"x")
    result = verify_compiled_dataset(root)
    assert result["valid"] is False
    assert any("token_size" in failure or "digest" in failure for failure in result["failures"])
    with pytest.raises(ValueError, match="verification failed"):
        CompiledDatasetSplit(root, "train")


def test_compiler_embeds_audit_and_rejects_cross_split_duplicates(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicates.csv"
    csv_path.write_text(
        "id,split,system,user,assistant\n"
        "1,train,Be exact,same prompt,same answer\n"
        "2,val,Be exact,same prompt,same answer\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cross-split"):
        compile_conversation_dataset(
            csv_path,
            output_dir=tmp_path / "rejected",
            tokenizer=ByteTokenizer(),
            split_spec=SplitSpec(train=0.5, val=0.5, test=0.0, respect_declared=True),
        )

    manifest_path = compile_conversation_dataset(
        csv_path,
        output_dir=tmp_path / "allowed",
        tokenizer=ByteTokenizer(),
        split_spec=SplitSpec(train=0.5, val=0.5, test=0.0, respect_declared=True),
        allow_cross_split_duplicates=True,
    )
    manifest = CompiledDatasetManifest.load(manifest_path)
    assert manifest.format_version == 2
    assert manifest.audit["cross_split_duplicate_rows"] == 1
    assert manifest.audit["accepted_rows"] == 2
