# WAI-R0 v0.4.4 — Chat CSV Schema Patch

This patch fixes the exact 500k-row instruction CSV shape shown by the user.

## Added

- Auto-detection for chat/instruction CSVs with columns:
  - `system`
  - `user`
  - `assistant`
  - optional `split`
- Safe fallback when old GUI defaults pass `--text-column text` against a chat CSV.
- Declared split support: `train`, `val`/`validation`/`dev`, and `test` are respected before hash-based fallback.
- GUI defaults now leave text/target columns blank so auto-detection works.
- Source-tree bootstrap in `main.py`, preserved from v0.4.3.

## Training text format

A row like:

```csv
system,user,assistant
"You are clear.","hello","Hello. What do you need help with?"
```

is converted internally to:

```text
SYSTEM:
You are clear.

USER:
hello

ASSISTANT:
Hello. What do you need help with?
```

This keeps the byte-level tokenizer simple while giving the tiny model a stable conversation envelope.

## Why this matters

Training only on the `assistant` column would teach response style but not the prompt/response relationship. Training only on `user` would teach prompts without answers. The combined chat format is the correct v0.4 path for this dataset.

## Still not claimed

This still does not make WAI-R0 a real chat model or an instruction-following LLM. It is a local language-readiness probe over a structured instruction CSV.
