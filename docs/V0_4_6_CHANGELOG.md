# WAI-R0 v0.4.6 — Hygiene and trust patch

This is a cleanup patch before v0.5. It does not claim new model intelligence.

## Fixed

- `python main.py` no longer crashes in headless shells when Tkinter cannot create a window. It falls back to a terminal workbench with direct commands.
- README and CSV docs now consistently state the split behavior: hash split is the default; declared CSV splits are opt-in with `--respect-csv-split`.

## Added

- `python main.py doctor` for local readiness checks.
- GUI label showing chat-schema auto-detection and hash-split default.
- Tests for doctor and GUI fallback behavior.

## Why

Before v0.5 adds task-family metrics and stronger evaluation, the repo needs trustworthy launch behavior, clearer split semantics, and a quick diagnostic command for local training machines.
