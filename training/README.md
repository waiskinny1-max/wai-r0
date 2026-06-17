# Training data directory

This directory is for local WAI-R0 training/probe inputs.

Use it for files such as a 500k-line CSV that teaches basic language patterns. Large datasets are ignored by `.gitignore`; do not commit them unless they are deliberately tiny examples.

## Supported CSV shapes

Single-column language rows:

```csv
text
The cat is on the mat.
A noun can name a person, place, or thing.
```

Prompt/completion rows:

```csv
prompt,completion
What is a noun?,A word that names a person, place, thing, or idea.
Complete: The sky is,blue.
```

Column autodetection prefers:

- text columns: `text`, `content`, `sentence`, `sample`;
- prompt columns: `prompt`, `instruction`, `input`, `question`, `source`;
- target columns: `completion`, `response`, `output`, `answer`, `target`.

Pass `--text-column` and `--target-column` when the CSV uses other names.

## Inspect first

```bash
wai-r0 inspect-csv --csv training/basic_lang_500k.csv --text-column text --sample-rows 1000
```

## Train a small CSV language probe

```bash
wai-r0 train-csv \
  --csv training/basic_lang_500k.csv \
  --text-column text \
  --config configs/model/nano.yaml \
  --steps 200 \
  --batch-size 8 \
  --seq-len 64 \
  --max-rows 500000 \
  --checkpoint checkpoints/csv_language_probe.pt \
  --output reports/csv_language_probe
```

Or use the Markdown-plan route:

```bash
python main.py -train training/csv_language_probe.md
```

## Scientific boundary

This is byte-level next-token training. A lower loss means the architecture learned some byte/text statistics under the chosen budget. It does not prove semantic understanding, reasoning, or AGI.
