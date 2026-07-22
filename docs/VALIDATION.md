# WAI-R0 v0.6 Validation Record

This record describes checks executed against the final local v0.6 implementation before the changed-files archive was assembled. It does not claim that unexecuted hardware, operating systems, or GitHub workflows passed.

## Environment

- Linux x86_64, glibc 2.41
- Python 3.13.5
- PyTorch 2.10.0+cpu
- Ruff 0.15.21
- mypy 2.2.0
- pytest 9.0.2
- build 1.5.1
- CUDA unavailable
- host PyTorch defaults: 56 intra-op and 28 inter-op threads

CPU training, evaluation, and inference tests used scoped `cpu_threads: 1` where appropriate. The process default was restored after each operation.

## Static quality, tests, and coverage

Final commands:

```bash
python scripts/check_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
```

Result before packaging:

- Ruff format passed;
- Ruff lint passed;
- mypy passed over 72 native source files;
- 135 tests passed;
- branch-aware native coverage was 80.90% over 6,403 statements and 2,032 branches, above the enforced 80% floor.

The count above includes v0.5 and v0.6 tests present in the exact applied payload. Preserved compressed v0.4 compatibility modules remain explicitly excluded from the native static-analysis/coverage denominator and are not a location for new v0.6 code.

## Executed end-to-end learning lineage

A deterministic CPU lineage was executed using 480 canonical conversation rows across copy, sorting, addition, and classification families:

1. train deterministic BPE;
2. compile and verify format-2 memory-mapped shards;
3. train a two-layer GQA model;
4. save digest-protected checkpoint format 3;
5. resume exactly from step 120 to step 140 while adding parent lineage;
6. run full validation-split language evaluation;
7. generate through the same chat template used in training.

### Data and tokenizer

- requested/actual tokenizer vocabulary: 384;
- compiled rows: 384 train, 48 validation, 48 test;
- rejected rows: 0;
- exact cross-split duplicates: 0;
- compiled dataset format: 2;
- validation split: 117 supervised target tokens and 255 target UTF-8 bytes.

The audit also reported 186 heuristic near-duplicate rows. This is expected from the deliberately templated synthetic corpus and is one reason these results are not presented as broad language generalization.

### Model

- vocabulary: 384;
- width: 64;
- layers: 2;
- query heads: 4;
- KV heads: 2;
- feed-forward width: 160;
- maximum sequence length: 96;
- attention: GQA;
- dtype/device: CPU float32;
- training mode: deterministic, packed, one intra-op thread.

### Learning result

- initial training loss: 6.0059919357;
- first scheduled validation loss: 3.1983609994;
- validation loss at step 120: 1.0454283754;
- exact resume completed at step 140;
- resumed progress: 280 consumed packed examples and 2,722 supervised target tokens;
- checkpoint lineage recorded the step-120 final checkpoint as the parent.

Full validation-split evaluation after resume:

- mean NLL: 0.9813453336;
- perplexity: 2.6680432368;
- bits per target token: 1.4157820462;
- bits per supervised target UTF-8 byte: 0.6495941153.

The bits-per-byte denominator is the supervised target text, not the entire prompt row. The report also retains total raw row bytes separately.

### Generation evidence

Observed deterministic examples after resume:

- `Case 1: Repeat beta twice.` → `beta beta`;
- `Sort these numbers: 8 2 5 1` → repetitive `222222222222`;
- `Add 4 and 28` → `22`.

The copy output is correct, while the sorting and arithmetic outputs are wrong. The honest interpretation is that the model learned narrow corpus patterns and generation mechanics; it did not demonstrate general sorting, arithmetic, or general reasoning.

## Sweep and registry checks

A six-trial recurrent-depth grid was planned with stable plan/trial hashes. One bounded trial was executed:

- executed trials: 1;
- failed trials: 0;
- outcome: `re_test`;
- the maximum-trial ceiling was enforced.

The resumed language report and checkpoint were inserted into the SQLite run registry and retrieved through the compact run-list surface.

## Packaging and release checks

The following completed successfully:

```bash
python -m compileall -q src
python scripts/verify_release.py --repository .
python -m build
```

The release doctor returned `ready: true`; git-related checks were warnings/unknown because the local reconstruction had no `.git` directory. Source distribution and wheel `0.6.0` built successfully. The wheel was installed into an isolated target and passed import, `python -m wai_r0 version`, installed `wai-r0 version`, and installed help-surface checks.

The final patch ZIP receives an internal SHA-256 manifest and is checked for caches, checkpoints, databases, generated reports, wheels, and build products.

## Not executed locally

- CUDA kernels and actual GPU allocator peaks;
- BF16/FP16 GPU training and scaler behavior;
- target 8 GB VRAM calibration;
- Windows and macOS jobs;
- real GitHub Actions runs on the pushed default-branch commit;
- the full live repository through a direct clone, because DNS blocked cloning in this environment.

The exact applied v0.5 payload was used as the local baseline and the live GitHub commit was inspected through the connector. Workflow definitions are included, but a workflow file is not evidence that GitHub executed it.
