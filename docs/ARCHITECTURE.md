# WAI-R0 v0.6 Architecture

## Objective

WAI-R0 separates model mechanics, data identity, training state, evaluation, and artifact lineage so an architecture comparison cannot silently change another experimental variable.

## Layers

1. **Modeling** — decoder, MHA/GQA/MLA-lite attention, cache, MoE, recurrence, and typed outputs.
2. **Tokenization** — deterministic byte control and deterministic byte-level BPE artifacts.
3. **Data** — CSV audit, chat encoding, compiled shards, packing, splits, and exact sampler state.
4. **Training** — optimizer, schedule, precision, accumulation, checkpoint lineage, validation, and telemetry.
5. **Evaluation** — algorithmic, language, context, generation, profiling, gates, and paired statistics.
6. **Operations** — hardware inventory/calibration, run registry, sweep execution, reports, and release verification.

## Stable application boundary

`wai_r0.app.cli` is the native command surface. `main.py`, `python -m wai_r0`, and the installed `wai-r0` script route to it. `wai_r0.v05_cli` remains a compatibility shim. Unknown legacy commands may delegate to the preserved v0.4 CLI, but new functionality must not be added there.

## Model contract

`ReasonerCore` retains tensor-return compatibility and supports structured `ModelOutput` containing logits, hidden state, cache, auxiliary losses, and opt-in diagnostics. Standard training avoids diagnostic host synchronizations.

Attention supports explicit positions, padding masks, packed block-diagonal masks, cached offsets, and cache metadata validation. MHA/GQA cache rotated K/V; MLA-lite caches compressed latent state. RoPE tables are cached by device, dtype, dimension, base, and range.

Recurrent steps are passed per call. Fixed, drift, and learned halt policies are experimental and must be compared against equal-compute controls. MoE reports total/active parameters, accepted and dropped routes, load, balancing loss, and router z-loss.

## Tokenizer and chat contract

Training and inference share one versioned chat template. The tokenizer artifact includes its vocabulary/merges, normalization, special tokens, corpus hash, and artifact hash. Byte fallback guarantees encodability. Any tokenizer or template change changes run identity and invalidates exact resume.

## Compiled data contract

Compilation audits source rows before writing memory-mapped token, label, and index shards. The manifest binds source, tokenizer, template, split, shard checksums, sample counts, target tokens, and raw bytes. Rejected rows or exact cross-split duplicates fail by default.

The compiled stream uses a deterministic affine permutation per epoch and stores epoch/cursor state. Packing preserves segment boundaries and prevents cross-example targets.

## Training and lineage

Checkpoint v3 stores model, optimizer, scheduler, scaler, progress, RNG, data state, resolved config, hashes, parent checkpoint, stage, tokenizer identity, and dataset identity. A SHA-256 sidecar protects integrity. Resume fails closed on incompatible scientific state.

## Evidence contract

Experiment and sweep manifests declare hypotheses, controls, matching rules, budgets, metrics, seeds, and thresholds. Correctness, provenance, seed-count, and matching gates are non-compensatory. Reports must distinguish learned language, learned algorithmic, systems, symbolic, hybrid, and diagnostic evidence.
