# v0.6 Quality Gates

## Local commands

```bash
python scripts/check_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python scripts/verify_release.py --repository .
python -m build
```

## Release floor

- Ruff format and lint pass.
- Mypy passes over the native package.
- All tests pass.
- Branch-aware native coverage is at least 80%.
- Checkpoint corruption and incompatible-resume tests pass.
- Tokenizer determinism/reference-equivalence tests pass.
- Compiled shard digest, source identity, split audit, and exact cursor tests pass.
- Source distribution and wheel build and import in isolation.
- `python main.py`, `python -m wai_r0`, and installed `wai-r0` agree on version and help.
- Patch/release artifacts contain no caches, checkpoints, local databases, reports, or build products.

## Compatibility denominator

Preserved compressed v0.4 compatibility modules remain explicitly excluded from native static-analysis/coverage denominators in `pyproject.toml`. They are not a location for new features. The release plan is to migrate useful behavior and shrink this exclusion rather than hide new code inside it.

## CI truth

Workflow files are not proof that CI ran. A release requires visible passing status contexts on the actual default-branch commit. CUDA claims additionally require a real GPU workflow or a documented target-machine run.
