# Bounded Sweeps

Sweep YAML defines a base experiment manifest and a finite grid of dotted-path overrides. Planning produces stable plan/trial hashes and materialized trial manifests. Execution is sequential by default and respects an explicit maximum-trial ceiling.

```bash
wai-r0 experiment sweep-plan configs/sweeps/recurrent_depth.yaml --output-dir reports/sweep
wai-r0 experiment sweep-run configs/sweeps/recurrent_depth.yaml --output-dir reports/sweep --maximum-trials 4
```

Sweeps do not replace hypotheses. Use them only when every trial contributes to a declared comparison and never tune on a frozen final test set.
