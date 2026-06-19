# WAI-R0 v0.4.5 — CSV split safety and checkpoint sampling fix

This patch fixes two issues surfaced by the live GUI training run.

## Fixed

- `sample-csv` no longer crashes with an opaque `FileNotFoundError` when the GUI points at a missing `reports/csv_probe.best.pt`. It now resolves common checkpoint naming mistakes and reports nearby `.pt` files when no checkpoint exists.
- The GUI sample tab now defaults to `reports/csv_probe.pt`, which is always written by a completed training run when `--checkpoint reports/csv_probe.pt` is used.
- CSV language training no longer silently falls back to evaluating on training rows when validation is empty.
- CSVs with a `split` column are now hash-split by default unless `--respect-csv-split` is explicitly passed. This prevents train-only CSVs from producing fake-perfect validation metrics.

## Added

- `--respect-csv-split` for `audit-csv` and `train-csv`.
- `--allow-train-eval-fallback` for intentional smoke tests only.
- Report fields: `schema` and `split_mode`.
- Clear audit warnings for train-only declared split columns.

## Interpretation

The previous live run completed, but its perfect validation accuracy was not trustworthy until split handling was hardened. A real run should show nonzero held-out loss and should not reach perfect validation accuracy instantly unless the validation set is trivial or duplicated.
