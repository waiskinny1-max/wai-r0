# WAI-R0 v0.4.3 — `python main.py` source-tree bootstrap

This patch fixes direct execution from a fresh source checkout.

## Fixed

- `python main.py` no longer fails with `ModuleNotFoundError: No module named 'wai_r0'` when the project has not been installed with `pip install -e .`.
- `main.py` now prepends the repository `src/` directory to `sys.path` before importing `wai_r0.cli`.
- The Tkinter workbench still falls back to the terminal workbench when no display is available.

## Why

WAI-R0 uses a standard Python `src/` layout. That layout is correct, but Python does not automatically import from `src/` when running a root-level script directly. The executable wrapper now performs that one local bootstrap step.

## Expected launch paths

```bash
python main.py
python main.py --help
python main.py train-csv --csv training/basic_language_sample.csv --text-column text --stream
python -m wai_r0 --help        # still works when PYTHONPATH=src or after editable install
wai-r0 --help                  # still works after editable install
```
