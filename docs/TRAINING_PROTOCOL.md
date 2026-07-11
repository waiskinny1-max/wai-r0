# WAI-R0 v0.5 Training Protocol

## Purpose

This protocol defines how a local training run becomes an auditable research artifact. It is stricter than a smoke test and intentionally narrower than production pretraining.

## Preflight

Before training:

1. run `wai-r0 doctor`;
2. validate the model configuration;
3. audit and manifest the dataset;
4. verify that the model vocabulary covers the tokenizer vocabulary;
5. verify that sequence length does not exceed `max_seq_len`;
6. declare the budget, objective, split seed, shuffle seed, and checkpoint interval;
7. profile the exact model and sequence length on the target device;
8. declare the CPU intra-op thread count when the run is CPU-bound.

Training must not start when the audit rejects malformed/duplicate-ID rows, when exact content crosses assigned splits, or when the requested validation split is empty.

## Objective

Assistant-only causal language modeling is the default conversation objective. System/user/role/padding tokens receive label `-100`. Full-sequence language modeling is a separate experiment and must be explicit.

For packed sequences, every segment receives block-diagonal causal attention. The first label in each segment is ignored so causal shifting never trains across example boundaries.

## Data order

The native CSV trainer uses a bounded deterministic shuffle by default. The shuffle buffer:

- has a declared capacity;
- uses a declared seed;
- never loads the complete dataset into memory;
- does not mix multiple epochs during initial fill;
- is serialized in checkpoints.

Set `--shuffle-buffer-size 0` only for a deliberate fixed-order control. Validation is fixed-order and unpacked by default.

## Optimization

Every run records:

- optimizer and hyperparameters;
- schedule and warmup;
- gradient accumulation;
- gradient clipping threshold;
- requested/effective precision;
- optimizer steps and microsteps;
- supervised target tokens;
- examples consumed;
- step time and target-token throughput.

A target-token budget is preferable when comparing configurations with different packing or target density. Step-matched comparisons are valid only when the manifest says so.


## CPU thread policy

PyTorch's process-wide intra-op thread default can be counterproductive for tiny local models: the dispatch and synchronization cost may exceed the matrix work. CPU runs may therefore set `cpu_threads` or `--cpu-threads`. WAI-R0 applies that setting only around the training/evaluation operation and restores the prior process value in `finally`, including on errors.

The thread count is part of the trainer configuration, checkpoint compatibility check, report, and reproduction command. Changing it during resume is rejected because it can alter performance and floating-point reduction behavior. WAI-R0 intentionally does not mutate inter-op thread count because PyTorch cannot safely reconfigure it after parallel work has begun.

## Mixed precision

- FP16 training requires CUDA and uses a gradient scaler.
- BF16 uses autocast without pretending the parameter dtype changed.
- CPU FP16 fails explicitly.
- A candidate that is stable only under a different precision than its control is not a clean architecture comparison.

## Validation

Validation uses an independently constructed deterministic stream. Report validation loss at declared intervals. Do not select a checkpoint using the final test set.

## Checkpoint and resume

Checkpoints include model, optimizer, scheduler, scaler, progress, RNG, and complete data-stream state. Digest sidecars are required by the native resume path unless the user explicitly opts out for a trusted local file.

Exact CPU resume means the restored run produces the same next batch, loss, optimizer transition, and parameter state under deterministic settings. GPU reproducibility is constrained by the selected kernels and runtime.

Changing resume-critical fields—learning rate, schedule, accumulation, precision, model mode, recurrence depth, or data semantics—fails closed. Extending a maximum budget is allowed but should be treated as an explicit continuation phase.

## Failure handling

- NaN/Inf loss or gradients: stop and mark correctness failure.
- OOM: fail with a dedicated diagnostic; do not silently reduce the batch or sequence length.
- Corrupt/mismatched digest: refuse to load.
- Keyboard interrupt: write an interrupt checkpoint when configured.
- Empty target batch: fail rather than stepping on a zero-supervision batch.

## Required artifacts

A native CSV run writes:

- `dataset-manifest.json`;
- `tokenizer-manifest.json`;
- structured `events.jsonl`;
- digest-protected checkpoint(s);
- `report.json`;
- `report.md`;
- `report.html`;
- `REPRODUCE.txt`.
