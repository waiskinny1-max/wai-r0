# WAI-R0

WAI-R0 is a zero-training reasoning architecture lab.

It does **not** claim random neural networks can reason. It tests whether a reasoning-oriented architecture has measurable structure worth training: stable signal propagation, memory behavior, recurrent latent refinement, MoE routing health, symbolic search compatibility, verifier integration, and tiny-training sample efficiency.

Status: `v0.4.1 local CSV language-readiness prototype`.

## What is implemented

- Decoder-only causal transformer with RMSNorm, RoPE, SwiGLU, MHA/GQA, deterministic initialization, and generation smoke path.
- MLA-lite compressed K/V attention. This is an MLA-inspired diagnostic module, not a DeepSeek MLA reproduction.
- Recurrent latent iterative refinement state with norm/drift logging. It is not called "thought" in reports.
- Tiny top-k MoE layer with router entropy, expert load, and collapse warning.
- ARC-style symbolic DSL and verified program search. Symbolic success is reported as symbolic, not neural.
- Tier-1 architecture-prior diagnostics for position, identity-signal, memory mechanics, recurrence, and MoE routing.
- Local leakage guard that hashes task content and flags cross-split duplicates.
- Deterministic generated ARC-style holdouts for local dev/validation separation.
- Multi-seed ablation matrix including A7 symbolic-only and A8 hybrid variants.
- Tiny-training probes with length extrapolation checks.
- JSON and markdown report export with metadata, limitations, and conservative recommendations.
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
wai-r0 architecture-priors --config configs/model/nano.yaml --seq-len 16 --recurrent-depths 1,2,4
wai-r0 memory --baseline mha --candidate mla_lite --seq-lens 64,128,256
wai-r0 symbolic-arc --tasks examples/tasks --budget 3s --leakage-manifest reports/leakage_manifest.json --split dev
wai-r0 generate-holdout --output-dir examples/generated_holdouts --count 8 --seed 2026
wai-r0 leakage-check --tasks examples/generated_holdouts --split generated_holdout --manifest reports/leakage_manifest.json --register
wai-r0 tiny-train --task copy --model configs/model/nano.yaml --examples 32 --train-len 8 --eval-lens 8,16,32
wai-r0 train examples/training.md
python main.py -train examples/training.md
wai-r0 ablate --matrix configs/benchmark/ablation.yaml --seeds 1337,2026 --tiny-examples 8
wai-r0 suite --config configs/model/nano.yaml --suite configs/benchmark/suite.yaml
wai-r0 report --input reports/latest.json --format md
python main.py  # opens the local Tkinter workbench
```

Use larger tiny-training budgets only after the smoke path works on your hardware.

## Result labels

| Label | Meaning |
|---|---|
| `zero-training neural diagnostic` | Random-weight numerical sanity. Not intelligence. |
| `architecture-prior diagnostic` | No-gradient architecture mechanics: activation stability, position proxy, identity signal, memory mechanics, recurrence, and routing. |
| `zero-training symbolic solver result` | Explicit symbolic program search. Not neural reasoning. |
| `tiny-training architecture probe` | Small supervised algorithmic learning probe. |
| `mixed architecture diagnostic` | Ablation report combining diagnostics; still not proof of reasoning. |



## v0.4.2 workbench fallback fix

`python main.py` still opens the local Tkinter workbench when a desktop display is available. If Tkinter is missing or cannot connect to a display, it now falls back to a terminal workbench instead of crashing. In non-interactive shells, it prints direct command examples and exits cleanly. See `docs/V0_4_2_CHANGELOG.md` and `docs/TKINTER_WORKBENCH.md`.

## v0.4.1 GUI update

`python main.py` opens a local Tkinter workbench when no CLI arguments are supplied and a display is available. The GUI lets you select a CSV file, audit it, launch CSV training with streamed logs, stop the run, sample from a checkpoint, and trigger the main benchmark commands. The GUI runs the same CLI commands underneath; it does not create a separate training path. See `docs/TKINTER_WORKBENCH.md`.

CLI streaming was also added:

```bash
wai-r0 train-csv --csv training/basic_lang_500k.csv --text-column text --stream
wai-r0 sample-csv --checkpoint reports/csv_probe.best.pt --prompt "A noun is" --stream
```

## v0.3 additions

| Area | What changed | Why it matters |
|---|---|---|
| Tier-1 diagnostics | Adds `architecture-priors` command | Tests architecture mechanics before wasting tiny-training runs. |
| Prior probes | Activation sanity, position proxy, identity signal, memory mechanics, recurrence, routing | Gives falsifiable no-gradient signals without claiming reasoning. |
| Suite runner | Adds `suite` command and `configs/benchmark/suite.yaml` | Runs the standard smoke ladder from one config. |
| Reporting | Prior suite exports JSON/Markdown through the existing report system | Keeps metadata, limitations, and recommendation discipline consistent. |

## v0.2.1 mini patch

`wai-r0 train <plan.md>` and `python main.py -train <plan.md>` load a Markdown training plan and run the existing tiny-training architecture probe. The Markdown file is declarative only: unsupported keys are rejected, and arbitrary Markdown/shell/Python instructions are not executed.

Supported plan fields:

```yaml
mode: tiny_probe
config: configs/model/nano.yaml
task: copy
examples: 16
batch_size: 4
train_len: 8
eval_lens: [8, 16]
output: reports/train_md
```

## v0.2 additions

| Area | What changed | Why it matters |
|---|---|---|
| Leakage guard | Hashes tasks and records split provenance | Avoids silently mixing generated/dev/public tasks. |
| Generated holdouts | Creates deterministic toy ARC-style tasks | Gives local validation tasks without pretending they are ARC-AGI. |
| Ablations | Adds A7 symbolic-only and A8 hybrid variants | Separates symbolic-system performance from neural diagnostics. |
| Tiny training | Reports evaluation accuracy across lengths | Starts measuring length extrapolation instead of only loss movement. |
| CLI | Adds `generate-holdout` and `leakage-check` | Makes the protocol runnable, not just documented. |

## Scientific limits

- Random weights do not contain learned language, world knowledge, arithmetic procedures, or planning skill.
- Symbolic solver results are system results, not neural-network reasoning.
- Tiny-training probes do not prove frontier reasoning.
- Generated holdouts are toy architecture diagnostics, not a substitute for ARC-AGI.
- ARC-style tasks are useful but incomplete and leakage-prone if repeatedly tuned against public eval.

## Repository map

```text
src/wai_r0/
  config.py              model/benchmark dataclasses and YAML loading
  model.py               transformer, attention, MLA-lite, MoE, recurrence, core API
  symbolic.py            ARC-style grid DSL and program search
  benchmarks.py          zero-neural, architecture-prior, memory, symbolic, tiny-train, ablations
  report.py              metadata, JSON/markdown reports, recommendations
  cli.py                 wai-r0 command surface
  eval/
    leakage_guard.py     content-hash task provenance checks
    prior_diagnostics.py architecture-prior probes
    suite.py             ordered diagnostic suite runner
    holdout.py           deterministic generated ARC-style tasks
    scorecard.py         keep/kill/re-test score helpers
  training/
    probes.py            tiny supervised algorithmic probes
    markdown_plan.py     declarative Markdown training-plan parser
```

## Current recommendation

v0.3 is good enough for local architecture iteration and honest diagnostic reports. It is **not** enough to justify serious pretraining. The next experiment should add broader algorithmic probes, longer-context profiler runs, baseline-vs-candidate plots, and stricter per-component keep/kill thresholds.
