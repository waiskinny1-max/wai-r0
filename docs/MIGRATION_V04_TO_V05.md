# Migrating v0.4.6 to v0.5

## Apply the overlay

Extract the changed-files archive at the repository root. It replaces selected files and adds new modules; it does not delete the v0.4 compatibility implementation.

## Command entrypoint

The package script now points to `wai_r0.v05_cli:main`. Native commands use nested forms such as:

```bash
wai-r0 train csv data.csv --config configs/model/nano.yaml --output-dir reports/run --max-steps 100
```

Existing names such as `train-csv`, `architecture-priors`, and `suite` are delegated to `wai_r0.cli`.

## Model API

Existing:

```python
logits = ReasonerCore(config)(tokens)
```

Structured:

```python
output = ReasonerCore(config)(tokens, use_cache=True, return_dict=True)
```

`think(tokens, budget)` no longer mutates configuration. The runtime state contains cache and mask semantics, not a dead scratchpad tensor.

## Configuration

Unknown fields now fail validation. Head dimension must be even for RoPE. MHA requires equal query/KV head counts. CPU FP16 is rejected. New MoE, recurrence, RoPE, deterministic, and diagnostic fields have conservative defaults.

## Checkpoints

v0.5 writes format 2 with model/config hashes and a `.sha256` sidecar. Native resume requires the sidecar by default. Legacy format 1 remains readable when structurally compatible.

## CSV training

The v0.5 native path adds strict audit, assistant-only targets, bounded deterministic shuffle, optional packed sequences, exact stream-state resume, and complete artifacts. It is separate from the old `train-csv` compatibility command.

## CI

All tests run across the matrix. Static analysis and coverage are scoped to v0.5 files until compressed v0.4 compatibility modules are migrated. See `docs/QUALITY_GATES.md`.
