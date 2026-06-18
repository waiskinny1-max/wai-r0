# Chat CSV schema

WAI-R0 v0.4.4 supports instruction-style CSV files with this shape:

```csv
id,split,task_family,difficulty,system,user,assistant,answer_format,eval_type,metadata_json
```

The required training columns are:

| Column | Purpose |
|---|---|
| `system` | Optional instruction/context text. |
| `user` | Prompt/query/task text. |
| `assistant` | Target answer text. |
| `split` | Optional declared split: `train`, `val`, `validation`, `dev`, or `test`. |

## Recommended GUI settings

For this schema, leave both fields blank:

```text
Text column:   blank
Target column: blank
```

The trainer will auto-detect `user` and `assistant` and will include `system` automatically.

## CLI audit

```bash
python main.py audit-csv \
  --csv training/synthetic_conversation_reasoning_500k.csv \
  --max-rows 500000 \
  --output reports/chat_csv_audit.json
```

## CLI train

```bash
python main.py train-csv \
  --csv training/synthetic_conversation_reasoning_500k.csv \
  --steps 500 \
  --batch-size 8 \
  --seq-len 128 \
  --max-rows 500000 \
  --eval-rows 256 \
  --eval-interval 25 \
  --baseline-rows 2048 \
  --checkpoint reports/csv_probe.pt \
  --log reports/csv_probe_train.jsonl \
  --output reports/csv_language_readiness \
  --stream
```

## Explicit override

Auto-detection is recommended. If you want to force columns manually:

```bash
python main.py train-csv \
  --csv training/synthetic_conversation_reasoning_500k.csv \
  --text-column user \
  --target-column assistant
```

Do not set `--text-column assistant` unless you intentionally want response-only language modeling.
