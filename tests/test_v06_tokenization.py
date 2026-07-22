from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from wai_r0.tokenization import (
    ByteTokenizer,
    DeterministicBPETokenizer,
    TokenizerTrainingConfig,
    train_bpe_from_conversation_csv,
    train_deterministic_bpe,
)
from wai_r0.tokenization.io import load_tokenizer


def test_byte_tokenizer_roundtrip_and_manifest() -> None:
    tokenizer = ByteTokenizer()
    text = "héllo — مرحبا"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    manifest = tokenizer.manifest()
    assert manifest["vocabulary_size"] == 261
    assert len(manifest["manifest_hash"]) == 64


def test_deterministic_bpe_training_is_reproducible_and_roundtrips(tmp_path: Path) -> None:
    corpus = ["banana bandana", "banana", "bandana", "banana"]
    first, first_summary = train_deterministic_bpe(corpus, vocab_size=280)
    second, second_summary = train_deterministic_bpe(corpus, vocab_size=280)
    assert first.merges == second.merges
    assert first_summary.corpus_hash == second_summary.corpus_hash
    sample = "banana bandana"
    encoded = first.encode(sample)
    assert len(encoded) < len(sample.encode("utf-8"))
    assert first.decode(encoded) == sample

    artifact = tmp_path / "tokenizer.json"
    first.save(artifact)
    loaded = DeterministicBPETokenizer.load(artifact)
    assert loaded.merges == first.merges
    assert loaded.decode(loaded.encode(sample)) == sample
    assert load_tokenizer(artifact).manifest() == loaded.manifest()


def test_bpe_training_from_csv_writes_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "chat.csv"
    csv_path.write_text(
        "id,system,user,assistant\n"
        "1,Be useful,hello world,hello back\n"
        "2,Be useful,hello again,hello there\n",
        encoding="utf-8",
    )
    output = tmp_path / "tokenizer.json"
    result = train_bpe_from_conversation_csv(
        csv_path,
        output=output,
        config=TokenizerTrainingConfig(vocab_size=270, max_training_bytes=10_000),
    )
    assert output.is_file()
    assert Path(result.summary_path).is_file()
    assert result.summary.actual_vocab_size >= 261
    assert len(result.manifest_hash) == 64


def test_bpe_rejects_invalid_merge_reference() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        DeterministicBPETokenizer([(999, 1)])


def test_chat_prompt_uses_training_role_template() -> None:
    from wai_r0.data.chat import encode_chat_prompt
    from wai_r0.tokenization import ByteTokenizer

    tokenizer = ByteTokenizer()
    prompt = encode_chat_prompt(
        "hello",
        system="Be concise.",
        tokenizer=tokenizer,
    )
    assert prompt[0] == tokenizer.bos_token_id
    assert tokenizer.system_token_id in prompt
    assert tokenizer.user_token_id in prompt
    assert prompt[-1] == tokenizer.assistant_token_id
    assert prompt.count(tokenizer.eos_token_id) == 2


def test_heap_bpe_matches_ranked_reference_semantics() -> None:
    from wai_r0.tokenization.bpe import DeterministicBPETokenizer

    merges = [(97, 98), (261, 99), (97, 97), (263, 97), (98, 99)]
    tokenizer = DeterministicBPETokenizer(merges)

    def reference(text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        ranks = {pair: rank for rank, pair in enumerate(merges)}
        while len(tokens) >= 2:
            available = [(ranks[pair], pair) for pair in pairwise(tokens) if pair in ranks]
            if not available:
                return tokens
            _rank, selected = min(available)
            replacement = 261 + ranks[selected]
            merged: list[int] = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == selected:
                    merged.append(replacement)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    for text in ["abc", "abcabc", "aaaaa", "zabcaaabc", "", "bcbcbc"]:
        assert tokenizer.encode(text) == reference(text)
