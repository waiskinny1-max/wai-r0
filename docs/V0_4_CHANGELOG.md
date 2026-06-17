# WAI-R0 v0.4 changelog

v0.4 is the first real training/evaluation milestone. Earlier CSV support only proved that rows could be streamed into a tiny byte-level objective. The final v0.4 release adds a controlled CSV language-readiness harness with corpus audit, deterministic splits, baselines, held-out evaluation, checkpoint/resume, and training logs.

## Added

- `/training/` directory with data conventions, ignore rules, sample CSV, and a CSV training plan.
- `wai-r0 inspect-csv` for fast schema and sample inspection.
- `wai-r0 audit-csv` for full/limited streaming audits:
  - detected columns;
  - nonempty row count;
  - duplicate row count/rate;
  - UTF-8 byte/character statistics;
  - deterministic train/validation/test split counts.
- `wai-r0 train-csv` for a held-out CSV language-readiness experiment:
  - stable hash-based data splits;
  - byte-level tokenizer;
  - uniform and byte-unigram baselines;
  - held-out validation loss and token accuracy;
  - JSON/Markdown report output;
  - JSONL training history;
  - checkpoint and best-checkpoint writing;
  - checkpoint resume.
- `wai-r0 sample-csv` to greedily inspect a CSV checkpoint output.
- `python main.py -train training.csv` compatibility for CSV shorthand.
- Markdown `mode: csv_language` plans now support the full v0.4 options.

## Scientific limit

v0.4 still does not claim language understanding. It measures whether WAI-R0 can reduce byte-level next-token loss on a user-provided CSV under a tiny controlled budget and whether that result beats trivial baselines on held-out rows.

The correct interpretation is:

```text
loss ↓ on validation + beats byte-unigram baseline = architecture/training path worth larger controlled tests
loss ↓ but does not beat byte-unigram = weak learning signal, re-test
no validation loss improvement = do not scale yet
```

## Recommended 500k-line flow

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

## Not added

- No BPE/SentencePiece tokenizer.
- No multi-GPU or distributed training.
- No instruction-following eval.
- No real ARC-AGI/language reasoning benchmark.
- No claim that a lower CSV loss equals reasoning.
