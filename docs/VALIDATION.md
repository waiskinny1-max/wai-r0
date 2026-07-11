# WAI-R0 v0.5 Validation Record

This record describes checks executed in the build environment before the changed-files archive was produced. It is not evidence that unexecuted hardware or operating systems passed locally.

## Environment

- Linux x86_64, glibc 2.41
- Python 3.13.5
- PyTorch 2.10.0+cpu
- Ruff 0.15.21
- mypy 2.2.0
- pytest 9.0.2
- build 1.5.1
- CUDA unavailable
- host PyTorch defaults: 56 intra-op threads, 28 inter-op threads

## Static quality

```bash
python scripts/check_v05_quality.py
```

Result: Ruff format, Ruff lint, and mypy passed across 44 v0.5 source files and the v0.5 test/quality files. Preserved compressed v0.4 compatibility modules are intentionally outside the v0.5 static-analysis denominator and remain inside the overlaid repository's runtime test suite.

## Tests and coverage

```bash
pytest -q
pytest --cov=wai_r0 --cov-report=term-missing
```

Result: 109 v0.5 tests passed. Branch-aware coverage was 81.54% across 4,201 measured source statements, above the enforced 80% floor.

The exact live `tests/test_model.py` compatibility surface fetched from `main` was reconstructed separately against the patch; all four tests passed. A complete clone of the live repository was unavailable in this execution environment, so the full original test suite could not be executed here.

## Executable experiment checks

### MLA-lite versus GQA cache experiment

```bash
PYTHONPATH=src python -m wai_r0.v05_cli experiment run \
  configs/experiments/mla_memory.yaml \
  --output /tmp/mla-v05.json \
  --render both
```

Observed in this CPU environment:

- three paired seeds completed;
- cached/full-context correctness gate passed;
- mean measured KV-cache reduction: 0.6821705426;
- preregistered decision: `keep` for further controlled study;
- this is a tiny CPU systems result, not a scale-transfer or GPU-latency claim.

### Recurrent refinement versus fast-mode control

```bash
PYTHONPATH=src python -m wai_r0.v05_cli experiment run \
  configs/experiments/recurrent_ood.yaml \
  --output /tmp/recurrent-v05.json \
  --render both
```

Observed in this CPU environment:

- three paired seeds completed;
- correctness and parameter-matching gates passed;
- mean held-out-length token-accuracy difference: 0.0096153846;
- 95% paired difference interval crossed zero;
- preregistered decision: `re_test` rather than `keep`;
- the candidate added active compute and was parameter-matched, not FLOP-matched.

The manifest uses `cpu_threads: 1`. With the host default of 56 threads, these tiny operations suffered severe threading overhead; the scoped setting reduced the full recurrent experiment to seconds while preserving/restoring the process default. This is why CPU thread policy is now explicit experiment state.

## Native training/resume smoke

A shuffled and packed CSV training run produced dataset/tokenizer manifests, events, JSON/Markdown/HTML reports, a format-2 checkpoint, and SHA-256 sidecar. Resume restored format-3 stream state, including shuffle RNG/buffer and pending epoch boundary, and continued the supervised target-token counter without resetting data order.

## Package checks

The source distribution and wheel were built, installed into an isolated target directory, imported, and reported version `0.5.0`. Packaging checks are rerun after the final source changes before archive creation.

## Unexecuted locally

- CUDA kernels, actual GPU peak allocation, BF16/FP16 GPU training, and 8 GB VRAM behavior;
- Windows and macOS jobs defined in GitHub Actions;
- the complete live v0.4.6 test suite, because the execution environment could inspect GitHub files but could not clone the repository;
- distributed training or scale-transfer claims, which are outside v0.5 scope.
