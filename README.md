# WAI-R0

**Current release:** `0.5.0` — Evidence Engine

WAI-R0 is a local-first research platform for testing language-model architecture ideas under explicit controls before committing to expensive training. Version 0.5 replaces the prototype model core and adds an auditable path from dataset to checkpoint to candidate/control report.

WAI-R0 does **not** claim that random weights reason, that symbolic solver performance is neural capability, or that tiny-model results transfer to frontier scale.

## v0.5 capabilities

### Correct model mechanics

- Modular decoder with RMSNorm, RoPE, SwiGLU, MHA/GQA, and MLA-lite.
- Real autoregressive MHA/GQA KV caching and MLA-lite latent caching.
- Padding, explicit positions, packed block-diagonal attention, and cache-aware masks.
- Cached/full-context and cached/uncached generation equivalence tests.
- Requested dtype applied to parameters; unsupported CPU FP16 fails explicitly.
- Explicit `ModelOutput` with cache, hidden states, auxiliary losses, and opt-in diagnostics.
- Recurrent refinement with per-call budgets and fixed/drift/learned stopping policies.
- Capacity-limited top-k MoE with accepted/dropped routes, balancing loss, router z-loss, and shared expert.

### Reproducible data and training

- Strict canonical conversation CSV audit, duplicate detection, deterministic splits, and manifests.
- Deterministic byte-chat tokenizer and assistant-only targets.
- Bounded deterministic shuffle whose buffer, RNG, and epoch boundary are checkpointed.
- Optional sequence packing with block-diagonal attention and boundary-target protection.
- AdamW, warmup, constant/linear/cosine schedules, accumulation, clipping, AMP, validation, and target-token telemetry.
- Step or supervised-target-token budgets.
- Atomic format-2 checkpoints with model/config hashes, exact stream state, RNG, and SHA-256 sidecars.

### Evidence engine

- Deterministic multi-family algorithmic probes with held-out-length evaluation.
- Preregistered candidate/control manifests.
- Paired seed statistics, bootstrap confidence intervals, and effect sizes.
- Non-compensatory correctness, matching, seed-count, robustness, and threshold gates.
- Measured prefill/decode/cache profiling with explicit, scoped CPU thread control.
- Versioned JSON plus deterministic Markdown and static HTML reports.
- Native v0.5 CLI while preserving v0.4 command delegation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
wai-r0 doctor
```

Python 3.10 or newer and PyTorch 2.2 or newer are required.

## Core commands

```bash
wai-r0 version
wai-r0 doctor
wai-r0 config validate configs/model/nano.yaml
wai-r0 data audit training/synthetic_conversation_reasoning_500k.csv --output reports/data-audit.json
wai-r0 data manifest training/synthetic_conversation_reasoning_500k.csv --output reports/data-manifest.json
wai-r0 model inspect --config configs/model/nano.yaml --seq-len 16 --diagnostics
wai-r0 profile --config configs/model/nano.yaml --seq-len 32 --cpu-threads 1 --output reports/profile.json
wai-r0 experiment validate configs/experiments/mla_memory.yaml
wai-r0 experiment run configs/experiments/mla_memory.yaml --output reports/mla-memory.json
wai-r0 report render reports/mla-memory.json --output reports/mla-memory.html
wai-r0 checkpoint inspect checkpoints/step-00000100.pt
```

Native conversation training:

```bash
wai-r0 train csv training/synthetic_conversation_reasoning_500k.csv \
  --config configs/model/mini_8gb.yaml \
  --output-dir reports/language-run \
  --max-target-tokens 1000000 \
  --batch-size 2 \
  --seq-len 256 \
  --gradient-accumulation-steps 8 \
  --mixed-precision bf16 \
  --checkpoint-every 100 \
  --shuffle-buffer-size 2048 \
  --pack-sequences
```


For CPU experiments, set `--cpu-threads` explicitly. Tiny tensor operations can be slower with dozens of intra-op threads; WAI-R0 scopes the requested value to the run, records it, and restores the process default afterward. The supplied CPU-facing experiment manifests use one intra-op thread.

Use `--max-steps` instead of `--max-target-tokens` for an explicitly step-matched run. Existing v0.4 names such as `architecture-priors`, `memory`, `suite`, `tiny-train`, `train-csv`, `generate-holdout`, `leakage-check`, and `ablate` are delegated to the compatibility CLI.

## Model API

The tensor-returning API remains compatible:

```python
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore

config = ReasonerConfig.from_yaml("configs/model/nano.yaml")
model = ReasonerCore(config)
tokens = torch.randint(0, config.vocab_size, (1, 8))
logits = model(tokens)
```

Structured output:

```python
output = model(tokens, use_cache=True, return_dict=True)
print(output.logits.shape)
print(output.past_key_values)
print(output.auxiliary_losses)
```

## Native run artifacts

A successful native CSV run writes:

- dataset and tokenizer manifests;
- structured event log;
- digest-protected checkpoint(s);
- JSON/Markdown/HTML report;
- exact `REPRODUCE.txt` command.

The stream state includes source hash, row/epoch cursor, shuffle contents/RNG, pending epoch boundary, and packing/objective semantics. Resume rejects changed source or incompatible configuration.

## 8 GB GPU profile

`configs/model/mini_8gb.yaml` is a conservative starting point, not a universal guarantee. Profile the actual machine first. Sequence length, resident batch, optimizer, precision, driver/runtime, MoE storage, and activation behavior materially change memory use.

## Verification

```bash
python scripts/check_v05_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python -m build
```

Static quality and the coverage floor apply to v0.5 files. Preserved v0.4 compatibility modules remain in the full runtime test suite. See `docs/QUALITY_GATES.md`.

## Scientific boundary

- Tiny generated-task evidence is screening evidence, not scale-transfer proof.
- MLA-lite is MLA-inspired and is not DeepSeek MLA.
- Recurrent latent states are not described as thoughts.
- Symbolic results remain symbolic or hybrid evidence.
- Estimates and measurements are labeled separately.
- CPU performance ordering may not transfer to GPU.
- Eight-gigabyte hardware cannot validate 7B pretraining claims.

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/TRAINING_PROTOCOL.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DATA_SCHEMA.md`
- `docs/TOKENIZATION.md`
- `docs/CHECKPOINT_FORMAT.md`
- `docs/EXPERIMENT_MANIFEST.md`
- `docs/HARDWARE_8GB.md`
- `docs/SCIENTIFIC_LIMITS.md`
- `docs/MIGRATION_V04_TO_V05.md`
- `docs/QUALITY_GATES.md`
- `docs/VALIDATION.md`
