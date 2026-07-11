# Checkpoint Format v2

WAI-R0 v0.5 checkpoints are trusted local PyTorch files.

| Field | Meaning |
|---|---|
| `format_version` | Schema integer; v0.5 writes `2` and reads formats `1` and `2`. |
| `wai_r0_version` | Package version that wrote the checkpoint. |
| `model_signature` | Canonical hash of model state names, shapes, and dtypes. |
| `model` | Model state dictionary. |
| `optimizer` | Optimizer state or `null`. |
| `scheduler` | Scheduler state or `null`. |
| `scaler` | AMP scaler state or `null`. |
| `progress` | Steps, microsteps, tokens, examples, epoch, cursor, elapsed time, and best metrics. |
| `config` / `config_hash` | Resolved configuration and canonical integrity hash. |
| `metadata` | Caller-supplied provenance. |
| `data_state` | Exact state of a stateful batch source. |
| `extra_state` | Additional caller-owned state. |
| `rng` | Python, CPU Torch, and CUDA RNG states. |

## CSV stream state v3

The native stream stores source hash, split/tokenizer semantics, row/epoch counters, bounded shuffle-buffer contents, shuffle RNG, pending epoch boundary, and packing mode. Formats 1 and 2 are readable only when newer shuffle/packing semantics are not requested.

## Atomicity and durability

The writer serializes into the destination directory, flushes and fsyncs the temporary file, atomically replaces the destination, fsyncs the directory where supported, then writes the SHA-256 sidecar through the same atomic process. Disabling digest generation removes a stale sidecar.

## Resume validation

Load validates:

- digest when required;
- supported format;
- model signature;
- model state;
- optimizer/scheduler/scaler presence when requested;
- progress structure;
- RNG type;
- configuration hash;
- mapping types for metadata/data/extra state.

The trainer additionally compares model and resume-critical trainer configuration before continuing.

## Security

Do not load untrusted checkpoint files. Optimizer and RNG restoration requires Python object deserialization. A checksum is an integrity signal, not a sandbox or signature.
