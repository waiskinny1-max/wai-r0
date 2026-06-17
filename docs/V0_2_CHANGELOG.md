# v0.2 Patch Notes

v0.2 expands WAI-R0 from a first runnable prototype into a stricter evaluation harness.

## Added

- Local leakage guard with content hashes and split-aware duplicate detection.
- Deterministic generated ARC-style holdout tasks.
- A7 symbolic-only and A8 hybrid ablation variants.
- Multi-seed ablation execution.
- Tiny-training length extrapolation fields.
- R0 scorecard helper for keep / kill / re-test framing.
- CLI commands for holdout generation and leakage checks.

## Changed

- `tiny-train` now reports token accuracy at configured evaluation lengths.
- `symbolic-arc` can attach leakage metadata to reports.
- `ablate` now evaluates neural, symbolic-only, and hybrid variants.
- README and Makefile document the v0.2 command surface.

## Still not claimed

- Random weights do not reason.
- Symbolic solver success is not neural reasoning.
- Tiny-training probes are not frontier reasoning evidence.
- Generated holdouts are toy diagnostics, not ARC-AGI results.
