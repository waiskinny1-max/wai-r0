# CSV language-readiness training

WAI-R0 v0.4 supports CSV-backed language-readiness experiments for local architecture work. The intended use case is a large synthetic or curated CSV, for example a 500k-line basic-language file.

This is not full language pretraining. It is a controlled tiny-training harness that answers one narrower question:

> Can this WAI-R0 architecture reduce held-out byte-level next-token loss on the CSV, and can it beat trivial byte baselines?


## Instruction/chat CSVs

For datasets shaped like:

```csv
id,split,task_family,difficulty,system,user,assistant,answer_format,eval_type,metadata_json
```

leave `--text-column` and `--target-column` unset. WAI-R0 will auto-detect `user` and `assistant`, include `system` when present, and respect the declared `split` column.

```bash
python main.py audit-csv --csv training/synthetic_conversation_reasoning_500k.csv --max-rows 500000
python main.py train-csv --csv training/synthetic_conversation_reasoning_500k.csv --steps 500 --batch-size 8 --seq-len 128 --stream
```

Use explicit columns only if auto-detection is wrong:

```bash
python main.py train-csv --csv training/synthetic_conversation_reasoning_500k.csv --text-column user --target-column assistant
```

## Supported CSV shapes

Single-text CSV:

```csv
text
A noun names a person, place, thing, or idea.
A verb names an action or state.
```

Prompt/completion CSV:

```csv
prompt,completion
What is a noun?,A noun names a person, place, thing, or idea.
What is a verb?,A verb names an action or state.
```

Detected text columns:

- `text`
- `content`
- `sentence`
- `sample`

Detected prompt columns:

- `prompt`
- `instruction`
- `input`
- `question`
- `source`

Detected target columns:

- `completion`
- `response`
- `output`
- `answer`
- `target`

For prompt/completion rows, WAI-R0 trains on:

```text
prompt + "\n" + completion
```

## Step 1 — Inspect quickly

```bash
wai-r0 inspect-csv \
  --csv training/basic_lang_500k.csv \
  --text-column text \
  --sample-rows 1000
```

This checks the header, detected columns, sample lengths, and obvious warnings.

## Step 2 — Audit the corpus

```bash
wai-r0 audit-csv \
  --csv training/basic_lang_500k.csv \
  --text-column text \
  --max-rows 500000 \
  --train-fraction 0.90 \
  --val-fraction 0.05 \
  --test-fraction 0.05 \
  --output reports/csv_audit.json
```

The audit streams rows and records:

- row counts;
- duplicate row count/rate;
- character and UTF-8 byte totals;
- deterministic split counts;
- warnings for empty splits or duplicate-heavy data.

## Step 3 — Run a real v0.4 probe

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

Outputs:

```text
reports/csv_language_readiness.json
reports/csv_language_readiness.md
reports/csv_probe.pt
reports/csv_probe.best.pt
reports/csv_probe_train.jsonl
```

## Step 4 — Resume if needed

```bash
wai-r0 train-csv \
  --csv training/basic_lang_500k.csv \
  --text-column text \
  --resume-from reports/csv_probe.pt \
  --steps 500 \
  --checkpoint reports/csv_probe_resume.pt \
  --log reports/csv_probe_resume.jsonl
```

## Step 5 — Inspect checkpoint output

```bash
wai-r0 sample-csv \
  --checkpoint reports/csv_probe.pt \
  --prompt "A noun is" \
  --max-new-tokens 80
```

This is a greedy byte-level sample. It is only an inspection tool. It is not a chat interface.

## Reading the result

Use the recommendation field:

| Result | Meaning |
|---|---|
| `KEEP` | Held-out loss improved and beat the byte-unigram baseline. Larger controlled tests are justified. |
| `RE-TEST` | Some learning signal exists, but the result is weak or the validation set is too small. |
| `KILL/REWORK` | Held-out loss did not improve under the budget. Do not scale this configuration. |

## Limits

- Tokenizer is byte-level and intentionally dependency-free.
- Splits are hash-based by row text, not document-aware.
- Duplicate templates can still leak structure across splits.
- Lower loss does not prove semantic understanding.
- No factuality, safety, instruction-following, or general reasoning evaluation is included yet.
