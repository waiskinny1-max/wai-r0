# Reproducibility

## Run identity

A reproducible run records package version, source commit/dirty state where available, resolved configuration, model signature, dataset manifest, tokenizer artifact, chat-template hash, seed, sampler state, hardware/software inventory, checkpoint lineage, and artifact checksums.

## Deterministic modes

CPU deterministic mode targets exact next-batch and parameter continuation where PyTorch supports it. GPU deterministic mode must record deterministic algorithms and backend choices. Throughput mode restores exact data/optimizer state but may permit declared numeric tolerance for nondeterministic kernels.

CPU intra-op threads are explicit experiment state for small models because excessive host threading can materially distort timing. Scoped settings are restored after an operation.

## Artifact verification

Compiled shards, tokenizer artifacts, checkpoints, reports, and packaged patch payloads carry hashes. `wai-r0 data verify`, `wai-r0 checkpoint inspect`, `wai-r0 release doctor`, and `scripts/verify_release.py` check available provenance.

## Reproduction

Prefer the exact command emitted by a run. For registered work:

```bash
wai-r0 runs show --database reports/runs.sqlite RUN_ID
```

For experiment reports:

```bash
wai-r0 reproduce reports/experiment.json
```

## Limits

Exact numerical identity is not promised across PyTorch, CUDA, driver, GPU, or operating-system changes. Wall-clock results are noisy. Missing source files prevent source rehashing. Trusted-local PyTorch checkpoints are not safe for untrusted input.
