# CSV language-probe training

WAI-R0 v0.4 supports CSV-backed language probes for local experiments. This is intended for files such as a 500k-line basic-language CSV.

## What it does

- streams rows from a CSV file;
- detects or accepts text/prompt/completion columns;
- tokenizes text at UTF-8 byte level;
- trains the existing `ReasonerCore` with a causal next-byte objective;
- exports JSON/Markdown reports;
- optionally writes a PyTorch checkpoint.

## What it does not do

- no BPE/SentencePiece tokenizer;
- no full language pretraining pipeline;
- no dataset license audit;
- no semantic reasoning benchmark;
- no claim that lower loss equals understanding.

## Recommended flow

1. Put the large file under `/training/`, for example `training/basic_lang_500k.csv`.
2. Inspect it:

```bash
wai-r0 inspect-csv --csv training/basic_lang_500k.csv --text-column text --sample-rows 1000
```

3. Run a small smoke train:

```bash
wai-r0 train-csv --csv training/basic_lang_500k.csv --text-column text --steps 25 --batch-size 4 --seq-len 64 --max-rows 1000
```

4. Increase budget only if loss moves and reports stay stable.

## CSV columns

The loader accepts either a direct text column or prompt/completion pairs.

Preferred names:

- direct text: `text`, `content`, `sentence`, `sample`;
- prompt: `prompt`, `instruction`, `input`, `question`, `source`;
- target: `completion`, `response`, `output`, `answer`, `target`.

For prompt/completion CSVs, the training row becomes `prompt + "\n" + completion`.
