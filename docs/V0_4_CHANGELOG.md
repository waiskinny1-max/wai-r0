# WAI-R0 v0.4 changelog

v0.4 adds a local CSV language-probe path for users who have large synthetic or curated language CSVs.

## Added

- `/training/` directory with data conventions, ignore rules, sample CSV, and Markdown training plan.
- `wai-r0 inspect-csv` to inspect schema, detected columns, sampled rows, and character lengths.
- `wai-r0 train-csv` to train a byte-level causal next-token probe from CSV rows.
- `python main.py -train training/csv_language_probe.md` dispatches to CSV mode when the Markdown plan declares `mode: csv_language`.
- Optional checkpoint writing with model config and byte-tokenizer metadata.

## Scientific limit

The CSV probe trains a tiny model on byte-level next-token prediction. It can measure whether loss drops under a controlled budget. It cannot establish semantic language understanding or reasoning.

## v0.4.1 compatibility note

The legacy shorthand now accepts CSV directly:

```bash
python main.py -train training/basic_language_sample.csv --steps 5 --max-rows 100
```

This is equivalent to the explicit CSV command:

```bash
wai-r0 train-csv --csv training/basic_language_sample.csv --steps 5 --max-rows 100
```

A `.csv` target is routed to the byte-level CSV language probe. A non-CSV target is routed to the Markdown training-plan runner.
