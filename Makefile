.PHONY: test zero memory symbolic tiny ablate

test:
	pytest
zero:
	PYTHONPATH=src python -m wai_r0 zero-neural --config configs/model/nano.yaml
memory:
	PYTHONPATH=src python -m wai_r0 memory --baseline mha --candidate mla_lite --seq-lens 64,128,256
symbolic:
	PYTHONPATH=src python -m wai_r0 symbolic-arc --tasks examples/tasks --budget 3s
tiny:
	PYTHONPATH=src python -m wai_r0 tiny-train --task copy --model configs/model/nano.yaml --examples 8
ablate:
	PYTHONPATH=src python -m wai_r0 ablate --matrix configs/benchmark/ablation.yaml
