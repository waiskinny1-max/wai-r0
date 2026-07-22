# WAI-R0 Data Contract

## Canonical conversation CSV

Recommended columns:

`id, split, task_family, difficulty, system, user, assistant, answer_format, eval_type, metadata_json`

`user` and `assistant` are required for native chat training. `id` should be stable and unique. `metadata_json`, when present, must decode to an object.

## Audit

The streaming audit checks UTF-8/CSV validity, field limits, metadata, duplicate IDs, exact normalized duplicate content, split assignment, cross-split duplication, and length distributions. Rejections and duplicate samples are recorded. Native compilation fails by default when any row is rejected.

Hash splitting remains the safe default. Declared splits are accepted only with an explicit option. Exact normalized content crossing train/validation/test fails compilation unless the user applies a recorded override.

## Compiled dataset format 2

Compilation produces a directory containing:

```text
manifest.json
train.tokens.bin
train.labels.bin
train.index.bin
val.tokens.bin
val.labels.bin
val.index.bin
test.tokens.bin
test.labels.bin
test.index.bin
```

Empty splits may omit shard files. The manifest records source and tokenizer hashes, chat-template identity, audit summary, split policy, shard checksums, sample/target-token totals, and raw UTF-8 byte totals. Verification rehashes shards and can rehash the original source when available.

Tokens and labels are memory mapped. Index records provide direct sample offsets. Labels use the training ignore index for context and padding. Assistant content is the supervised target by default.

## Exact iteration

Compiled iteration stores split, seed, epoch, cursor, and permutation semantics. The affine permutation is deterministic and O(1) in memory. Packing combines examples only when block boundaries and first-target masking prevent leakage.

## Limits

Exact hashes do not detect semantic paraphrases. Release datasets should add near-duplicate clustering and license/provenance review before public weight training.
