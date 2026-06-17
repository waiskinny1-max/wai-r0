# WAI-R0

WAI-R0 is a zero-training reasoning architecture lab.

It does **not** claim random neural networks can reason. It tests whether a reasoning-oriented architecture has measurable structure worth training: stable signal propagation, memory behavior, recurrent latent refinement, MoE routing health, symbolic search compatibility, verifier integration, and tiny-training sample efficiency.

Status: `v0.1 research prototype`.

## What is implemented

- Decoder-only causal transformer with RMSNorm, RoPE, SwiGLU, MHA/GQA, deterministic initialization, and generation smoke path.
- MLA-lite compressed K/V attention. This is an MLA-inspired diagnostic module, not a DeepSeek MLA reproduction.
- Recurrent latent iterative refinement state with norm/drift logging. It is not called "thought" in reports.
- Tiny top-k MoE layer with router entropy, expert load, and collapse warning.
- ARC-style symbolic DSL and verified program search. Symbolic success is reported as symbolic, not neural.
- CLI for zero-neural diagnostics, memory estimates, symbolic ARC tasks, tiny training, ablations, and report conversion.
- JSON and markdown report export with metadata, limitations, and conservative recommendation.
- CPU-safe tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Commands

```bash
wai-r0 zero-neural --config configs/model/nano.yaml
wai-r0 memory --baseline mha --candidate mla_lite --seq-lens 64,128,256
wai-r0 symbolic-arc --tasks examples/tasks --budget 3s
wai-r0 tiny-train --task copy --model configs/model/nano.yaml --examples 8
wai-r0 ablate --matrix configs/benchmark/ablation.yaml
wai-r0 report --input reports/latest.json --format md
```

Use larger tiny-training budgets only after the smoke path works on your hardware.

## Result labels

| Label | Meaning |
|---|---|
| `zero-training neural diagnostic` | Random-weight numerical sanity. Not intelligence. |
| `architecture-prior diagnostic` | No-gradient architecture mechanics. |
| `zero-training symbolic solver result` | Explicit symbolic program search. Not neural reasoning. |
| `tiny-training architecture probe` | Small supervised algorithmic learning probe. |

## Scientific limits

- Random weights do not contain learned language, world knowledge, arithmetic procedures, or planning skill.
- Symbolic solver results are system results, not neural-network reasoning.
- Tiny-training probes do not prove frontier reasoning.
- ARC-style tasks are useful but incomplete and leakage-prone if repeatedly tuned against public eval.

## Repository map

```text
src/wai_r0/
  config.py       model/benchmark dataclasses and YAML loading
  model.py        transformer, attention, MLA-lite, MoE, recurrence, core API
  symbolic.py     ARC-style grid DSL and program search
  benchmarks.py   zero-neural, memory, symbolic, tiny-train, ablations
  report.py       metadata, JSON/markdown reports, recommendations
  cli.py          wai-r0 command surface
```

## Current recommendation

This v0.1 is useful for local diagnostics and architecture iteration. It is **not** enough to justify serious pretraining until the tiny-training and ablation suites are expanded across seeds, lengths, generated holdouts, and hardware.
