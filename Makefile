.PHONY: test zero prior memory symbolic symbolic-holdout tiny ablate suite leakage holdout smoke

test:
	pytest

zero:
	PYTHONPATH=src python -m wai_r0 zero-neural --config configs/model/nano.yaml

memory:
	PYTHONPATH=src python -m wai_r0 memory --baseline mha --candidate mla_lite --seq-lens 64,128,256

prior:
	PYTHONPATH=src python -m wai_r0 architecture-priors --config configs/model/nano.yaml --seq-len 16 --recurrent-depths 1,2,4

symbolic:
	PYTHONPATH=src python -m wai_r0 symbolic-arc --tasks examples/tasks --budget 3s --leakage-manifest reports/leakage_manifest.json --split dev

symbolic-holdout:
	PYTHONPATH=src python -m wai_r0 symbolic-arc --tasks examples/generated_holdouts --budget 3s --leakage-manifest reports/leakage_manifest.json --split generated_holdout

tiny:
	PYTHONPATH=src python -m wai_r0 tiny-train --task copy --model configs/model/nano.yaml --examples 32 --train-len 8 --eval-lens 8,16,32

ablate:
	PYTHONPATH=src python -m wai_r0 ablate --matrix configs/benchmark/ablation.yaml --seeds 1337,2026 --tiny-examples 8

suite:
	PYTHONPATH=src python -m wai_r0 suite --config configs/model/nano.yaml --suite configs/benchmark/suite.yaml

leakage:
	PYTHONPATH=src python -m wai_r0 leakage-check --tasks examples/generated_holdouts --split generated_holdout --manifest reports/leakage_manifest.json --register

holdout:
	PYTHONPATH=src python -m wai_r0 generate-holdout --output-dir examples/generated_holdouts --count 8 --seed 2026

smoke: test zero prior memory symbolic tiny ablate
