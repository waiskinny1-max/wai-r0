# Compiled Datasets

Compiled datasets remove repeated CSV parsing and bind training to verified token/label/index shards.

## Build

```bash
wai-r0 data compile input.csv --tokenizer tokenizer.json --output-dir artifacts/dataset
```

The compiler audits the full source, encodes the versioned chat template, writes split shards atomically, computes checksums, and emits `manifest.json`. It fails on rejected rows and exact cross-split content by default.

## Verify

```bash
wai-r0 data verify artifacts/dataset
```

Shard verification is always possible while files exist. Source verification is possible when the original CSV remains at the recorded path.

## Iteration and resume

Index records point to memory-mapped token and label ranges. A deterministic affine permutation provides shuffling without storing a full permutation. State includes split, seed, epoch, and cursor. Packing is optional and preserves block boundaries and target masking.

Compiled data is not a license or privacy review. Public weight releases still need a provenance ledger and near-duplicate analysis.
