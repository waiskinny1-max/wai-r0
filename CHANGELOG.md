# Changelog

## 0.5.0 — Evidence Engine

### Architecture correctness

- Replaced the compressed prototype model with modular attention, cache, feed-forward/MoE, recurrence, output, and transformer components.
- Added real MHA/GQA cached decoding and MLA-lite latent caching.
- Added cache metadata validation, explicit position IDs, padding masks, and packed block-diagonal attention.
- Added cached/full-context and generation-equivalence invariants.
- Applied configured dtype to parameters and made invalid device/precision requests explicit.
- Removed recurrent frozen-config mutation and unused runtime scratchpad state.
- Moved diagnostics out of the ordinary hot path.

### MoE and recurrence

- Added top-k route normalization, capacity factor/minimum capacity, highest-weight acceptance, dropped-route accounting, raw/accepted loads, balancing loss, router z-loss, and shared expert.
- Added fixed, drift, and learned recurrent stopping with per-call budgets and optional ponder loss.
- Added total/trainable/active-per-token parameter accounting.

### Data and tokenization

- Added strict canonical conversation schema, streaming audit, rejection sampling, duplicate-ID/content checks, deterministic split assignment, and hash-verified manifests.
- Added deterministic byte-chat tokenizer manifests and assistant-only target masking.
- Added bounded deterministic shuffle with exact buffer/RNG/epoch-boundary restoration.
- Added optional greedy packing with block-diagonal causal masks and cross-example target protection.

### Training and checkpoints

- Added unified trainer with AdamW groups, constant/linear/cosine schedules, warmup, accumulation, clipping, AMP, validation, step and target-token budgets, and structured telemetry.
- Added format-2 atomic checkpoints containing model, optimizer, scheduler, scaler, progress, RNG, data state, model signature, and config hash.
- Added atomic SHA-256 sidecars, stale-sidecar removal, digest verification, and fail-closed resume compatibility checks.

### Evaluation and reporting

- Added deterministic algorithmic tasks: copy, reverse, parity, modular addition, sorting, selective copy, associative recall, bracket balance, and finite-state parity.
- Added in-distribution and held-out-length evaluation, paired-seed comparisons, bootstrap confidence intervals, effect sizes, wins/losses/ties, and failed-seed visibility.
- Added executable profile, algorithmic, and external-metric experiment manifests with preregistered thresholds and budget checks.
- Added non-compensatory gates and versioned JSON/Markdown/static-HTML reports.
- Added measured prefill/decode/cache/throughput profiling.
- Added scoped CPU intra-op thread control for training, evaluation, and profiling; the effective value is reported and restored after each operation.

### Tooling and compatibility

- Added native v0.5 CLI commands and preserved v0.4 command delegation.
- Added complete native CSV artifacts and reproduction commands.
- Added scoped Ruff/mypy gates, branch-aware coverage floor, cross-platform tests, package build, and clean-wheel smoke CI.
- Added migration, architecture, protocol, reproducibility, data, tokenizer, hardware, quality, security, and scientific-limit documentation.
