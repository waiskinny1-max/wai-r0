# Tkinter workbench

`v0.4.1` adds a local Tkinter workbench for WAI-R0. It is a control panel for experiments, not a polished product UI.

## Launch

From the repository root:

```bash
python main.py
```

Equivalent explicit command:

```bash
PYTHONPATH=src python -m wai_r0 gui
```

## What it does

- Select a CSV file from `training/` or anywhere on disk.
- Audit the CSV before training.
- Start CSV language-probe training with live streamed step logs.
- Stop a running training process.
- Watch step count, train loss, eval loss, and step speed.
- Pick a checkpoint and stream a greedy byte-level sample.
- Run the existing smoke benchmarks from buttons.

## What it does not do

- It does not make WAI-R0 a chat model.
- It does not hide the CLI commands; every action still runs the same CLI underneath.
- It does not replace the JSON/Markdown reports.
- It does not implement full language-model pretraining.

## Safe first run

```text
CSV: training/basic_lang_500k.csv
Config: configs/model/nano.yaml
Steps: 500
Batch: 8
Seq: 128
Eval rows: 256
Eval every: 25
Checkpoint: reports/csv_probe.pt
Log: reports/csv_probe_train.jsonl
Report stem: reports/csv_language_readiness
```

Use the **Audit CSV** button first. Then use **Start training**. Use **Talk / sample** only after a checkpoint exists.

## Streaming behavior

The CLI now supports:

```bash
PYTHONPATH=src python main.py train-csv --csv training/basic_lang_500k.csv --text-column text --stream
```

It prints compact progress events:

```text
[train] {"step": 1, "train_loss": 5.55, "eval_loss": 5.51, ...}
```

The GUI parses these lines and updates the progress readout.
