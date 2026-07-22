# 8 GB GPU Operating Profile

`configs/model/mini_8gb.yaml` is a conservative starting configuration. It is not a memory guarantee.

## Preflight sequence

1. Run `wai-r0 release doctor` and `wai-r0 hardware inspect`.
2. Run `wai-r0 hardware estimate` for a theoretical preflight.
3. Run `wai-r0 hardware calibrate` on the target GPU and record the result.
4. Start long training from the largest measured-safe profile below the configured memory fraction.
5. Increase gradient accumulation before resident batch size.
6. Enable packing only after objective and boundary tests pass.
7. Keep checkpoint and validation intervals large enough to avoid dominating short runs.

## Memory levers

In approximate order of impact:

- sequence length;
- resident batch size;
- layer count and hidden width;
- optimizer state;
- activation checkpointing when available;
- attention cache representation;
- MoE total parameter storage;
- precision.

Gradient accumulation reduces resident batch memory but does not reduce optimizer/model memory. MLA-lite reduces cache payload, not necessarily total training memory. MoE increases total parameter memory even when active compute is sparse.

## Precision

Use BF16 when the device/runtime supports it reliably. Otherwise use FP16 with scaling. Do not silently fall back to FP32 while reporting a lower precision.

## OOM policy

WAI-R0 reports OOM rather than silently changing semantics. Reduce one declared variable, create a new configuration/run identity, and rerun the profiler.


## CPU controls

Small-model CPU experiments should benchmark a low intra-op thread count rather than inherit every host core automatically. The supplied manifests use `cpu_threads: 1`; this is a reproducibility default for tiny probes, not a universal performance recommendation. Profile one, two, and a modest number of threads on the actual CPU before selecting a value for larger models. The profiler records the effective count and restores the previous process setting.
