# WAI-R0 v0.5 Architecture

## Design objective

WAI-R0 v0.5 is an evidence-first local research platform. The implementation is organized so an architecture candidate can be changed without silently changing the dataset, objective, training budget, evaluation protocol, or report interpretation.

The runtime is divided into six layers:

1. **Modeling** — decoder, attention/cache, recurrence, and MoE mechanics.
2. **Data** — schema validation, deterministic splits, deduplication, tokenization, packing, and stateful streaming.
3. **Training** — objective, optimizer, schedule, mixed precision, checkpointing, evaluation, and telemetry.
4. **Evaluation** — generated algorithmic tasks, exact/token metrics, profilers, and mandatory gates.
5. **Experiments** — preregistered candidate/control execution and paired statistics.
6. **Reporting** — versioned JSON plus deterministic Markdown and static HTML renderers.

The old v0.4 CLI and benchmark modules remain compatibility surfaces. New v0.5 commands are implemented in `wai_r0.v05_cli` and delegate unknown legacy command names to `wai_r0.cli` when that module exists.

## Model contract

`ReasonerCore` wraps `DecoderOnlyTransformer` and retains the v0.4 tensor-returning API. Structured execution returns `ModelOutput`:

- `logits`;
- optional `hidden_states`;
- optional `past_key_values`;
- named `auxiliary_losses`;
- opt-in diagnostics.

Diagnostics are deliberately outside the ordinary hot path. Standard training does not call `.item()` merely to populate human-readable telemetry.

### Attention

Three attention modes are supported:

- **MHA** — one K/V head per query head;
- **GQA** — fewer K/V heads repeated across query groups;
- **MLA-lite** — a compressed latent cache projected into K/V at use time.

MHA/GQA cache rotated keys and values. MLA-lite caches the latent representation, position IDs, and key-padding metadata, then reconstructs keys and values during decode. This is a controlled MLA-inspired mechanism, not a reproduction of a frontier implementation.

All attention paths support:

- explicit position IDs;
- left or right padding masks;
- block-diagonal packed-sequence masks;
- cached decode offsets;
- cache metadata validation;
- reference cache-equivalence tests.

### RoPE

RoPE tables are cached by device, dtype, dimension, base, and required position range. Position IDs are derived from a 2D key-padding mask when possible or supplied explicitly for packed sequences. Head dimensions must be even.

### Recurrent refinement

Recurrent refinement is an optional latent update after the decoder stack. The number of refinement steps is passed per call; configuration objects are never mutated. Supported stopping policies are:

- fixed depth;
- drift threshold;
- learned halting probability.

Adaptive halting is an experimental mode because stopping checks can synchronize device and host. It must be profiled separately from fixed-depth execution.

### MoE

The tiny MoE path includes:

- top-k routing;
- optional top-k weight normalization;
- per-expert capacity;
- highest-router-weight acceptance under capacity pressure;
- dropped-route accounting;
- raw and accepted expert load;
- load-balancing auxiliary loss;
- router z-loss;
- optional shared dense expert.

Total parameters and estimated active parameters are reported separately. An MoE comparison is invalid unless its manifest declares and satisfies a matching rule.

## Data contract

The canonical conversation schema supports:

`id, split, task_family, difficulty, system, user, assistant, answer_format, eval_type, metadata_json`

The audit pipeline streams the CSV, validates rows, records rejection samples, detects duplicate IDs/content, assigns deterministic splits, and writes a content-hashed manifest. Training fails closed when rows are rejected or exact content crosses assigned splits.

The byte tokenizer is a deterministic control with a stable manifest. Assistant-only loss is the default. Packed training uses block-diagonal causal attention and masks the first target at every segment boundary so shifting cannot create a cross-example target.

## Exact stream resume

`StatefulCSVBatchStream` persists:

- source hash;
- split policy;
- tokenizer-manifest hash;
- row and epoch cursors;
- examples and batches emitted;
- bounded shuffle-buffer contents;
- shuffle RNG state;
- pending epoch-boundary state;
- packing and objective semantics.

The shuffle implementation drains one epoch before starting the next. A buffer larger than the dataset does not preload duplicate examples from multiple epochs.

## Training contract

The trainer accepts any stateful batch source implementing `state_dict` and `load_state_dict`. It provides:

- AdamW with decay-safe parameter groups;
- constant, linear, or cosine schedules;
- warmup;
- gradient accumulation and clipping;
- FP16/BF16 autocast where supported;
- validation intervals;
- target-token and optimizer-step budgets;
- atomic checkpointing;
- exact data-state restoration;
- structured event callbacks.

A resumed run rejects changes to model or resume-critical trainer semantics. Output path and larger budget fields may change, but such extensions should be reported as a new run phase.

## Evidence contract

Experiment manifests separate hypothesis, candidate, control, matching rule, primary metric, seeds, thresholds, maximum budget, confounds, and final-evaluation status. The runner produces paired per-seed rows and uncertainty statistics.

Decisions are non-compensatory. A performance win cannot override failed correctness, successful-seed, contamination, or robustness gates.
