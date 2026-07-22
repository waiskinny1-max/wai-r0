# Migration from v0.5 to v0.6

## CLI

Use `wai_r0.app.cli` as the application entry point. `wai_r0.v05_cli` remains import-compatible but should not receive new code.

## Tokenizers

Existing byte-tokenized v0.5 checkpoints remain readable with their original tokenizer. A BPE tokenizer is a different model vocabulary and cannot be substituted into an existing checkpoint.

## Data

CSV remains an ingestion source. New long runs should compile it first and train from the compiled directory. Compilation performs stricter audit and may reject data that v0.5 accepted, particularly exact content crossing declared splits.

## Checkpoints

v0.6 writes checkpoint format 3 and can read prior formats where compatible. Format 3 adds parent/stage/tokenizer/dataset lineage. Resume of an old checkpoint requires explicitly compatible data and tokenizer semantics.

## Commands

Replace `python scripts/check_v05_quality.py` with `python scripts/check_quality.py`. The old script remains a compatibility shim.

## Reports and runs

Existing v0.5 reports remain historical artifacts. Register only reports whose schema and artifact references can be validated. Re-run release verification after overlaying the patch.
