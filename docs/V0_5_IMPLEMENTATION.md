# WAI-R0 v0.5 Implementation Record

## Delivered platform

Version 0.5 is an architecture, data, training, evaluation, reporting, and reproducibility reset built against the live v0.4.6 repository API.

Delivered components include:

- modular transformer/attention/cache/MoE/recurrence implementation;
- MHA/GQA real KV cache and MLA-lite latent cache;
- cache/full-context and generation equivalence tests;
- precision-correct construction and explicit device failures;
- structured model outputs and opt-in diagnostics;
- padding and block-diagonal packed attention;
- deterministic byte-chat objective with assistant-only masking;
- strict CSV schema/audit/deduplication/split manifest;
- bounded deterministic shuffle with exact epoch and checkpoint semantics;
- sequence packing with cross-example target protection;
- unified trainer with AMP, accumulation, schedules, validation, and target-token telemetry;
- digest-protected atomic full-state checkpoints and exact data resume;
- generated multi-family algorithmic battery;
- paired statistics, confidence intervals, and non-compensatory gates;
- measured local profiler with scoped/reported CPU thread policy;
- versioned JSON, Markdown, and static HTML reports;
- native v0.5 CLI plus v0.4 command delegation;
- cross-platform test CI, scoped static quality, package build, and clean-wheel smoke.

## Compatibility decisions

`wai_r0.model` remains a compatibility facade. Ordinary model calls still return logits. `ReasonerConfig` accepts old fields, including `latent_scratchpad_size`, but v0.5 does not expose an unused scratchpad tensor as runtime state.

`wai_r0.training` and `wai_r0.eval` lazily expose v0.4 symbols when their old modules are present. No-argument CLI execution delegates to the old workbench when available.

## Validation in the build environment

The final archive is validated immediately before packaging. `VALIDATION.md` records exact commands, counts, and environment limits. The target environment here is CPU-only; CUDA claims are therefore limited to code/invariant checks and CI definitions, not observed GPU execution.

## Deliberately deferred

The following remain later work rather than hidden partial implementations:

- distributed training;
- production inference service;
- large-corpus ingestion;
- reinforcement learning;
- speculative decoding;
- a browser UI;
- validated quantization matrix;
- 7B training claims;
- enforcement of FLOP-, wall-clock-, or memory-matched algorithmic controls beyond explicit runner failure.
