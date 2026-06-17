# WAI-R0 CSV language-probe plan

```yaml
mode: csv_language
config: configs/model/nano.yaml
csv_path: training/basic_language_sample.csv
text_column: text
steps: 5
batch_size: 2
seq_len: 32
max_rows: 10
lr: 0.0003
eval_rows: 4
checkpoint: checkpoints/csv_language_probe.pt
output: reports/csv_language_probe_from_md
```

This plan runs a tiny byte-level CSV language probe. It is not full pretraining.
