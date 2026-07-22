# WAI-R0

**Current release:** `0.6.0` — Ground Truth

WAI-R0 is a local-first research platform for training and evaluating small language-model architecture candidates under explicit controls. Version 0.6 extends the v0.5 Evidence Engine with a deterministic subword tokenizer, compiled memory-mapped datasets, checkpoint lineage, native inference and language evaluation, hardware calibration, run registration, bounded sweep execution, and release verification.

WAI-R0 does **not** treat tiny-model learning as proof of general reasoning, symbolic results as neural capability, estimates as measurements, or CPU results as GPU validation.

## What v0.6 adds

### Tokenization and compiled data

- Deterministic byte tokenizer remains the zero-dependency control.
- Deterministic byte-level BPE with fixed role tokens, byte fallback, corpus and artifact hashes, and round-trip validation.
- Versioned chat encoding shared by training and inference.
- Compiled memory-mapped token, label, and index shards with SHA-256 verification.
- Exact cross-split duplicate auditing enabled by default.
- Exact-resume affine shuffle state and optional packed sequence blocks.

### Training and lineage

- Native training from compiled shards.
- Checkpoint format 3 with parent checkpoint, training stage, tokenizer hash, dataset hash, full optimizer/scheduler/scaler/RNG state, and digest sidecars.
- Step or supervised-target-token budgets.
- Optional activation checkpointing, fused AdamW, and `torch.compile`.
- Telemetry for target/raw throughput, padding, gradient and parameter norms, scaler state, and GPU allocator peaks.

### Evaluation and inference

- Held-out NLL, perplexity, bits per target token, and bits per raw UTF-8 byte.
- Synthetic context retrieval and induction diagnostics.
- Generation diagnostics for repetition, diversity, longest runs, and EOS behavior.
- Greedy and seeded stochastic generation with temperature, top-k, top-p, min-p, repetition penalty, stop sequences, and cache reuse.

### Operations

- CPU/CUDA capability inventory, rough memory estimation, and target-device calibration.
- SQLite run registry with lineage and artifact records.
- Deterministic bounded grid sweeps with stable plan and trial hashes.
- Release doctor and clean-source verification.
- Stable native CLI at `wai_r0.app.cli`; legacy v0.4 commands remain a narrow compatibility fallback.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
wai-r0 version
wai-r0 release doctor
```

Python 3.10 or newer and PyTorch 2.2 or newer are required.

## Ground-truth workflow

### 1. Train a tokenizer

```bash
wai-r0 tokenizer train training/chat.csv \
  --output artifacts/tokenizer.json \
  --vocab-size 4096 \
  --max-training-bytes 50000000

wai-r0 tokenizer inspect artifacts/tokenizer.json
```

### 2. Compile and verify the dataset

```bash
wai-r0 data compile training/chat.csv \
  --tokenizer artifacts/tokenizer.json \
  --output-dir artifacts/datasets/chat-v1 \
  --respect-declared-split

wai-r0 data verify artifacts/datasets/chat-v1
```

Compilation fails by default when rows are rejected or exact normalized content crosses splits. Overrides are explicit and recorded.

### 3. Train from compiled shards

```bash
wai-r0 train compiled artifacts/datasets/chat-v1 \
  --tokenizer artifacts/tokenizer.json \
  --config configs/model/mini_8gb.yaml \
  --output-dir reports/mini-baseline \
  --max-target-tokens 1000000 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --pack-sequences \
  --mixed-precision bf16 \
  --checkpoint-every 100
```

Use `--resume-from checkpoints/final.pt` to continue. Resume validates model, trainer semantics, tokenizer, compiled dataset, and exact sampler state.

### 4. Evaluate and generate

```bash
wai-r0 eval language artifacts/datasets/chat-v1 \
  --tokenizer artifacts/tokenizer.json \
  --config configs/model/mini_8gb.yaml \
  --checkpoint reports/mini-baseline/checkpoints/final.pt \
  --split val

wai-r0 infer generate \
  --tokenizer artifacts/tokenizer.json \
  --config configs/model/mini_8gb.yaml \
  --checkpoint reports/mini-baseline/checkpoints/final.pt \
  --prompt "Explain the result." \
  --max-new-tokens 64
```

### 5. Register and compare work

```bash
wai-r0 runs init --database reports/runs.sqlite
wai-r0 runs register --database reports/runs.sqlite reports/mini-baseline/report.json
wai-r0 runs list --database reports/runs.sqlite
```

### 6. Plan controlled sweeps

```bash
wai-r0 experiment sweep-plan configs/sweeps/recurrent_depth.yaml \
  --output-dir reports/sweeps/recurrent-depth

wai-r0 experiment sweep-run configs/sweeps/recurrent_depth.yaml \
  --output-dir reports/sweeps/recurrent-depth \
  --maximum-trials 4
```

## Hardware calibration

```bash
wai-r0 hardware inspect
wai-r0 hardware estimate --config configs/model/mini_8gb.yaml --batch-size 1 --seq-len 512
wai-r0 hardware calibrate --config configs/model/mini_8gb.yaml --target-memory-fraction 0.90
```

The estimator is theoretical. Calibration records actual allocator behavior where CUDA exists. It does not silently rewrite an experiment.

## Model and experiment commands

```bash
wai-r0 config validate configs/model/nano.yaml
wai-r0 model inspect --config configs/model/nano.yaml --seq-len 16 --diagnostics
wai-r0 profile --config configs/model/nano.yaml --seq-len 64 --cpu-threads 1
wai-r0 experiment validate configs/experiments/mla_memory.yaml
wai-r0 experiment run configs/experiments/mla_memory.yaml --output reports/mla-memory.json
wai-r0 report render reports/mla-memory.json --output reports/mla-memory.html
wai-r0 reproduce reports/mla-memory.json
```

## Verification

```bash
python scripts/check_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python scripts/verify_release.py --repository .
python -m build
```

The current release floor is branch-aware coverage of at least 80% over the native code, with preserved legacy compatibility modules explicitly excluded from the denominator but still available to runtime tests.

## Scientific boundary

- A decreasing loss proves learning on the measured distribution, not general intelligence.
- A generated exact answer can be memorized or template-induced; held-out composition and contamination controls remain necessary.
- MLA-lite is MLA-inspired, not a reproduction of any frontier implementation.
- Recurrent latent state is not called thought.
- Symbolic or verifier-assisted success is reported as hybrid evidence.
- CUDA, mixed precision, and 8 GB claims require measurements on the target GPU.
- A small local model cannot justify 7B-scale conclusions without a scaling study.

## Documentation

- `docs/V0_6_IMPLEMENTATION.md`
- `docs/ARCHITECTURE.md`
- `docs/BPE_TOKENIZER.md`
- `docs/COMPILED_DATASETS.md`
- `docs/TRAINING_PROTOCOL.md`
- `docs/CHECKPOINT_FORMAT.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/INFERENCE.md`
- `docs/HARDWARE_CALIBRATION.md`
- `docs/RUN_REGISTRY.md`
- `docs/SWEEPS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/QUALITY_GATES.md`
- `docs/VALIDATION.md`
- `docs/MIGRATION_V05_TO_V06.md`
