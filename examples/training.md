# WAI-R0 tiny-training plan

This file is a declarative tiny-training probe config. It is not executable code
and it is not a pretraining recipe.

```yaml
mode: tiny_probe
config: configs/model/nano.yaml
task: copy
examples: 16
batch_size: 4
train_len: 8
eval_lens: [8, 16]
output: reports/train_md
```
