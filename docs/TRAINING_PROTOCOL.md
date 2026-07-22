# Training Protocol

## Required identity

A native run binds:

- resolved model and trainer configuration;
- source/compiled dataset manifest;
- tokenizer artifact and chat template;
- random seed and sampler state;
- software/hardware inventory;
- training stage and parent checkpoint.

## Budget

Use supervised target tokens as the primary learning budget when target density or packing differs. Step budgets are valid only for explicitly step-matched comparisons. Reports must include microbatches, optimizer steps, raw tokens, target tokens, examples, elapsed time, and throughput.

## Optimization

The trainer supports AdamW decay-safe groups, warmup, constant/linear/cosine schedules, accumulation, clipping, validation, FP16/BF16 autocast where supported, optional activation checkpointing, optional fused AdamW, and optional `torch.compile`. Performance flags must be recorded and compared against a reference path before being treated as equivalent.

## Validation and selection

Validation data must be independent and contamination audited. Best-checkpoint rules must be fixed before results. Training loss alone cannot promote an architecture. Preserve failed runs and non-finite diagnostics.

## Resume

Resume restores model, optimizer, scheduler, scaler, RNG, progress, and exact compiled-data cursor. Tokenizer and dataset hashes are release-critical. A continuation with changed data, template, objective, or precision policy must be a new run lineage.

## Target hardware

Run hardware calibration before long GPU training. OOM recovery must never silently alter batch size, sequence length, precision, or accumulation. The tool may recommend a safer config, but the resolved experiment must remain explicit.
