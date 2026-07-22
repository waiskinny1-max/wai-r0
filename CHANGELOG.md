# Changelog

## 0.6.0 — Ground Truth

### Repository and application surface

- Moved the native command implementation to the stable `wai_r0.app.cli` entry point and retained `v05_cli` only as a compatibility shim.
- Added `python -m wai_r0`, stable top-level help, a release doctor, and clean-source verification.
- Added pull-request, nightly, release, reproducibility, and optional self-hosted CUDA workflows.

### Tokenization and data

- Added deterministic byte-level BPE with byte fallback, role tokens, corpus/artifact hashes, save/load, and reference-equivalence tests.
- Kept deterministic byte tokenization as the zero-dependency control.
- Added one versioned chat encoding path shared by training and inference.
- Added format-2 compiled memory-mapped datasets with checksummed token/label/index shards, exact split summaries, source/tokenizer/template hashes, and raw-byte accounting.
- Added full pre-compilation audit and fail-closed exact cross-split duplicate rejection.
- Added deterministic O(1) affine shuffle state and optional packed sequence blocks.

### Training and checkpoints

- Added native compiled-dataset training and exact resume.
- Added checkpoint format 3 with parent lineage, training stage, tokenizer hash, dataset hash, and complete optimizer/scheduler/scaler/RNG/data state.
- Added optional activation checkpointing, fused AdamW, and `torch.compile` paths.
- Expanded telemetry with target/raw throughput, padding fraction, parameter norm, scaler scale, and GPU allocator peaks.

### Hardware, evaluation, inference, and registry

- Added CPU/CUDA capability inventory, theoretical memory estimation, and measured calibration.
- Added held-out language evaluation with NLL, perplexity, bits per target token, and bits per raw byte.
- Added synthetic context retrieval/induction evaluation and generation diagnostics.
- Added native cached generation with greedy and seeded sampling controls.
- Added a transactional SQLite run registry with lineage and artifact records.
- Added bounded deterministic grid sweeps with stable plan/trial hashes.

### Validation

- Added 0.6-specific unit, invariant, corruption, CLI, lineage, compilation, inference, sweep, and end-to-end tests.
- Updated static-quality, coverage, package, and release verification commands.
- Executed a real deterministic BPE → compiled dataset → training → exact resume → inference lineage on CPU. The model learned narrow held-out templates but did not demonstrate general arithmetic or classification reasoning.

## 0.5.0 — Evidence Engine

### Architecture correctness

- Replaced the compressed prototype model with modular attention, cache, feed-forward/MoE, recurrence, output, and transformer components.
- Added real MHA/GQA cached decoding and MLA-lite latent caching.
- Added cache metadata validation, explicit position IDs, padding masks, and packed block-diagonal attention.
- Added cached/full-context and generation-equivalence invariants.
- Applied configured dtype to parameters and made invalid device/precision requests explicit.
- Removed recurrent frozen-config mutation and unused runtime scratchpad state.
- Moved diagnostics out of the ordinary hot path.

### Data, training, evaluation, and tooling

- Added strict conversation audit, deterministic byte-chat training, shuffle/packing, exact checkpoints, algorithmic experiments, paired statistics, non-compensatory gates, reports, profiling, and native v0.5 CLI compatibility.
