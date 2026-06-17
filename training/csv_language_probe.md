---
mode: csv_language
config: configs/model/nano.yaml
csv_path: training/basic_language_sample.csv
text_column: text
steps: 25
batch_size: 4
seq_len: 64
max_rows: 1000
eval_rows: 16
eval_interval: 5
baseline_rows: 256
train_fraction: 0.90
val_fraction: 0.05
test_fraction: 0.05
split_seed: 1337
checkpoint: reports/csv_probe.pt
log: reports/csv_probe_train.jsonl
output: reports/csv_language_readiness
---

# CSV language-readiness probe

This is a declarative v0.4 training plan. It runs the same held-out CSV language-readiness path as `wai-r0 train-csv`.

It is not full language pretraining and not evidence of semantic reasoning. It tests whether the architecture can reduce held-out byte-level next-token loss and beat trivial byte baselines.
