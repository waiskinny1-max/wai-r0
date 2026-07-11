# Apply the WAI-R0 v0.5 Patch

This archive contains **only new or changed files**. It is an overlay, not a complete repository checkout.

## Apply

From the root of the existing `wai-r0` clone:

```bash
git switch -c v0.5-evidence-engine
unzip wai-r0-v0.5-live-audited-changed-files-only.zip -d .
git status --short
git diff --stat
git diff
```

Files with matching paths are intentional replacements. Files absent from the archive must remain in place because the v0.5 CLI delegates legacy command names to the preserved v0.4 implementation.

## Verify

```bash
python -m pip install -e ".[dev]"
python scripts/check_v05_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python -m build
wai-r0 version
wai-r0 doctor
wai-r0 config validate configs/model/nano.yaml
wai-r0 experiment validate configs/experiments/mla_memory.yaml
wai-r0 experiment validate configs/experiments/recurrent_ood.yaml
```

Expected package version: `0.5.0`.

## Review before commit

```bash
git diff --check
git status --short
git add -A
git diff --cached --stat
git commit -m "WAI-R0 v0.5 Evidence Engine"
```

## Hardware note

The supplied 8 GB configurations request CUDA. CUDA execution was not available in the build environment. Profile and smoke-test them on the target GPU before any long run.
