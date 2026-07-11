from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from wai_r0.config import BenchmarkConfig, ReasonerConfig


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"d_model": 0}, "positive"),
        ({"d_model": 18, "n_heads": 4}, "divisible"),
        ({"d_model": 12, "n_heads": 4, "n_kv_heads": 4}, "head_dim"),
        ({"n_heads": 4, "n_kv_heads": 3}, "n_kv_heads"),
        ({"attention_type": "mha", "n_heads": 4, "n_kv_heads": 2}, "mha"),
        (
            {"attention_type": "mla_lite", "n_kv_heads": 2, "mla_latent_dim": 32},
            "smaller",
        ),
        ({"dropout": 1.0}, "dropout"),
        ({"n_experts": 2, "experts_per_token": 3}, "experts_per_token"),
        ({"moe_capacity_factor": 0.0}, "capacity"),
        ({"moe_load_balance_coef": -1.0}, "coefficients"),
        ({"rope_base": 1.0}, "rope_base"),
        ({"norm_epsilon": 0.0}, "norm_epsilon"),
        ({"initialization_std": 0.0}, "initialization_std"),
        ({"recurrent_halt_mode": "fixed", "recurrent_halt_threshold": 0.1}, "only valid"),
        ({"recurrent_halt_mode": "drift", "recurrent_halt_threshold": None}, "requires"),
        ({"recurrent_halt_mode": "drift", "recurrent_halt_threshold": 0.0}, "positive"),
        ({"recurrent_halt_mode": "learned", "recurrent_halt_threshold": 1.0}, r"in \(0, 1\)"),
        ({"recurrent_depth": 1, "recurrent_min_steps": 2}, "cannot exceed"),
        ({"recurrent_ponder_loss_coef": -0.1}, "ponder"),
        ({"latent_scratchpad_size": -1}, "scratchpad"),
        ({"dtype": "int8"}, "dtype"),
        ({"device": ""}, "device"),
    ],
)
def test_reasoner_config_rejects_invalid_combinations(changes, message) -> None:
    base = ReasonerConfig()
    with pytest.raises(ValueError, match=message):
        replace(base, **changes).validate()


def test_reasoner_config_strict_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        ReasonerConfig.from_dict({"made_up": 1})
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        ReasonerConfig.from_yaml(path)


def test_benchmark_config_validation_and_yaml(tmp_path) -> None:
    valid = BenchmarkConfig()
    valid.validate()
    assert valid.to_dict()["name"] == "default"
    invalid = [
        replace(valid, name=""),
        replace(valid, seeds=[]),
        replace(valid, seeds=[1, 1]),
        replace(valid, seq_lens=[0]),
        replace(valid, batch_size=0),
        replace(valid, timeout_s=0),
        replace(valid, output_dir=""),
    ]
    for item in invalid:
        with pytest.raises(ValueError):
            item.validate()
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "seeds": [1]}), encoding="utf-8")
    assert BenchmarkConfig.from_yaml(path).name == "x"
    path.write_text(yaml.safe_dump({"unknown": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        BenchmarkConfig.from_yaml(path)
