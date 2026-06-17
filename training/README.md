# training/

Put local training data here. Large files are intentionally ignored by git.

## Expected large CSV

Use a CSV such as:

```csv
text
A noun names a person, place, thing, or idea.
A verb names an action or state.
```

or:

```csv
prompt,completion
What is a noun?,A noun names a person, place, thing, or idea.
```

## v0.4 flow

```bash
wai-r0 inspect-csv --csv training/basic_lang_500k.csv --text-column text
```

```bash
wai-r0 audit-csv \
  --csv training/basic_lang_500k.csv \
  --text-column text \
  --max-rows 500000 \
  --output reports/csv_audit.json
```

```bash
wai-r0 train-csv \
  --csv training/basic_lang_500k.csv \
  --text-column text \
  --steps 500 \
  --batch-size 16 \
  --seq-len 128 \
  --max-rows 500000 \
  --eval-rows 256 \
  --eval-interval 25 \
  --baseline-rows 2048 \
  --checkpoint reports/csv_probe.pt \
  --log reports/csv_probe_train.jsonl \
  --output reports/csv_language_readiness
```

## Ignored files

`training/.gitignore` ignores real training corpora and checkpoints. Keep only small examples in git.
