# Apply the WAI-R0 v0.6 Ground Truth Patch

This archive contains only files added or changed relative to the already-applied v0.5 Evidence Engine commit. It is an overlay, not a complete repository.

## Apply

From the repository root:

```bash
git switch -c v0.6-ground-truth
unzip wai-r0-v0.6-ground-truth-changed-files-only.zip -d .
git status --short
git diff --stat
git diff --check
```

Do not delete files absent from the archive. Legacy v0.4 modules remain as a narrow compatibility fallback.

## Verify

```bash
python -m pip install -e ".[dev]"
python scripts/check_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python scripts/verify_release.py --repository .
python -m build
python main.py version
python -m wai_r0 version
wai-r0 release doctor
wai-r0 tokenizer --help
wai-r0 data --help
wai-r0 train --help
```

Expected version: `0.6.0`.

## Commit

```bash
git add -A
git diff --cached --stat
git diff --cached --check
git commit -m "WAI-R0 v0.6 Ground Truth"
git push
```

## Hardware boundary

The implementation contains CUDA inspection and calibration, but the build environment used for this patch had CPU-only PyTorch. Run `wai-r0 hardware calibrate` on the target 8 GB GPU before a long mixed-precision run. Do not treat the theoretical estimator as a measurement.
