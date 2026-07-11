# WAI-R0 v0.5 Reproducibility

## Run identity

Every v0.5 report records:

- WAI-R0 version;
- UTC creation time;
- command arguments;
- resolved configuration and canonical hash;
- experiment hash when applicable;
- Git commit and dirty-tree status when available;
- hardware and software inventory;
- dataset and tokenizer manifests;
- artifact paths and failures.

A missing Git repository is recorded as unavailable, not invented.

## Determinism controls

`set_seed` initializes Python and Torch RNGs and can request deterministic Torch algorithms. Determinism is a property of the full environment, not only the seed. PyTorch version, CUDA runtime, device, dtype, kernels, and thread behavior can affect results. Native CPU training/profiling can declare an intra-op thread count; the value is recorded and scoped to the operation.

## Exact data continuation

The stateful CSV stream validates source content hash and all semantic settings before restoration. It restores the precise row cursor, epoch, shuffle-buffer contents, shuffle RNG, pending boundary state, and packing mode. A changed source file or incompatible stream setting fails closed.

## Checkpoint integrity

Checkpoint format 2 includes:

- a model-structure signature;
- a canonical configuration hash;
- atomic file replacement;
- fsync of file and containing directory where supported;
- an atomic SHA-256 sidecar.

The native trainer requires the digest on resume by default. This detects accidental corruption or substitution; it is not a cryptographic signature of author identity.

## Reproduction workflow

For native CSV training:

```bash
cat reports/run/REPRODUCE.txt
# inspect the command, source hashes, and configuration before executing it
```

For experiments:

```bash
wai-r0 experiment validate configs/experiments/example.yaml
wai-r0 experiment run configs/experiments/example.yaml --output reports/example.json
wai-r0 reproduce reports/example.json
```

`wai-r0 reproduce` verifies available provenance. `--execute` reruns only when the report contains a resolvable experiment manifest.

## Limits

- Exact results are not guaranteed across different PyTorch/CUDA/hardware combinations.
- CPU comparisons must declare the intra-op thread count; the host inter-op thread setting is inventoried but not mutated.
- Wall-clock metrics are inherently noisy.
- Python pickle-based checkpoints must be treated as trusted local artifacts.
- A report can verify hashes only while the referenced files remain available.
- Small generated tasks do not establish scale transfer.
