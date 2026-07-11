# v0.5 Quality Gates

## Commands

```bash
python scripts/check_v05_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python -m build
```

The quality script runs Ruff formatting/linting and mypy over the v0.5 implementation plus v0.5 tests. The complete pytest invocation also executes preserved legacy tests after the patch is overlaid onto the live repository.

## Compatibility scope

The repository still contains v0.4 compatibility modules. They are deliberately excluded from the v0.5 static-analysis and coverage denominator until migrated because changing their behavior is outside the architecture reset. They are not excluded from runtime tests.

Excluded compatibility modules are listed explicitly in `pyproject.toml`; new v0.5 modules are not excluded. This prevents an apparently green result created by ignoring newly written code while avoiding a false CI failure caused solely by untouched compressed legacy files.

## Release floor

- all compatibility and v0.5 tests pass;
- v0.5 Ruff and mypy gates pass;
- branch-aware v0.5 coverage is at least 80%;
- source and wheel build;
- clean wheel import and CLI smoke pass;
- archive contains no cache, local checkpoint, or build byproduct;
- archive manifest and SHA-256 are generated after final validation.
