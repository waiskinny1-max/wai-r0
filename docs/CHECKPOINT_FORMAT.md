# Checkpoint Format v3

WAI-R0 0.6 writes trusted-local PyTorch checkpoint format `3` and reads formats `1`, `2`, and `3` where compatible.

## Core fields

| Field | Meaning |
|---|---|
| `format_version` | Checkpoint schema version. |
| `wai_r0_version` | Package version that wrote the file. |
| `model_signature` | Hash of model state names, shapes, and dtypes. |
| `model` | Model parameters and buffers. |
| `optimizer` / `scheduler` / `scaler` | Complete training state or `null`. |
| `progress` | Steps, microsteps, examples, tokens, elapsed time, and best metrics. |
| `config` / `config_hash` | Resolved run configuration and canonical hash. |
| `data_state` | Exact state of the batch source/sampler. |
| `rng` | Python, CPU Torch, and CUDA RNG states. |
| `metadata` / `extra_state` | Validated caller-owned provenance. |

## v3 lineage

Format 3 adds explicit lineage metadata:

- `parent_checkpoint`;
- `training_stage`;
- `tokenizer_hash`;
- `dataset_hash`.

These values identify the parent and the exact tokenizer/dataset artifacts. A resumed run rejects incompatible lineage instead of silently continuing on different data.

## Durability

Checkpoint and SHA-256 sidecar are written through temporary files, flushed, fsynced where supported, and atomically replaced. Loading can require the digest. A checksum detects corruption or substitution but is not a signature and does not make pickle safe.

## Resume rules

Exact resume requires compatible model signature, resolved critical config, optimizer/scheduler/scaler presence, tokenizer hash, dataset hash, sampler state, and RNG. Output directories and larger terminal budgets may change only when the service explicitly treats the continuation as an extension phase.

## Security

Load only trusted local checkpoint files. Python/Torch deserialization can execute unsafe payloads. Publish checksums and provenance, but do not describe them as sandboxing.
