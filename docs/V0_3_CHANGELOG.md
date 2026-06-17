# v0.3 changelog

v0.3 adds a real Tier-1 architecture-prior diagnostics layer between zero-neural sanity checks and tiny training.

## Added

- `wai-r0 architecture-priors` command.
- `wai-r0 suite` command for running a small ordered diagnostic suite.
- No-gradient architecture-prior probes:
  - activation sanity;
  - positional addressing proxy;
  - identity-signal preservation proxy;
  - memory-mechanics comparison for MHA/GQA/MLA-lite;
  - recurrent-depth consistency;
  - MoE routing health.
- `configs/benchmark/prior.yaml`.
- `configs/benchmark/suite.yaml`.
- `scripts/run_architecture_priors.py`.
- `scripts/run_suite.py`.
- Tests for the prior diagnostics and suite runner.

## Scientific boundary

These probes do **not** show semantic reasoning. They measure mechanical architecture properties before training. A good score only means "worth further controlled testing," not "worth expensive pretraining."

## Recommendation

Use v0.3 before ablations:

```bash
wai-r0 architecture-priors --config configs/model/nano.yaml
wai-r0 suite --config configs/model/nano.yaml --suite configs/benchmark/suite.yaml
```

If architecture-prior probes fail before tiny training, kill or simplify the architecture before increasing training budget.
