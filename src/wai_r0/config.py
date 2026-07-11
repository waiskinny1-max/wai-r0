from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

AttentionType = Literal["mha", "gqa", "mla_lite"]
DTypeName = Literal["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]
RecurrentHaltMode = Literal["fixed", "drift", "learned"]


@dataclass(frozen=True, slots=True)
class ReasonerConfig:
    """Configuration for the compact decoder and experimental components.

    Defaults remain deliberately small so correctness checks run on CPU. New
    fields have conservative defaults and preserve all v0.4 configuration keys.
    """

    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 1
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 64
    max_seq_len: int = 64
    attention_type: AttentionType = "mha"

    rope_base: float = 10_000.0
    norm_epsilon: float = 1e-6
    initialization_std: float = 0.02
    use_sdpa: bool = True
    qkv_bias: bool = False

    use_moe: bool = False
    n_experts: int = 4
    experts_per_token: int = 1
    moe_capacity_factor: float = 1.25
    moe_min_capacity: int = 1
    moe_load_balance_coef: float = 0.01
    moe_router_z_loss_coef: float = 0.001
    moe_shared_expert: bool = True
    moe_normalize_topk: bool = True

    recurrent_depth: int = 1
    recurrent_halt_mode: RecurrentHaltMode = "fixed"
    recurrent_halt_threshold: float | None = None
    recurrent_min_steps: int = 1
    recurrent_ponder_loss_coef: float = 0.0
    latent_scratchpad_size: int = 8  # accepted for v0.4 config compatibility

    mla_latent_dim: int = 12
    dropout: float = 0.0
    tie_embeddings: bool = True
    dtype: DTypeName = "float32"
    device: str = "cpu"
    seed: int = 1337
    deterministic: bool = False
    diagnostics_default: bool = False

    def validate(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "n_experts": self.n_experts,
            "experts_per_token": self.experts_per_token,
            "recurrent_depth": self.recurrent_depth,
            "recurrent_min_steps": self.recurrent_min_steps,
            "mla_latent_dim": self.mla_latent_dim,
            "moe_min_capacity": self.moe_min_capacity,
        }
        invalid = [name for name, value in positive_ints.items() if value < 1]
        if invalid:
            raise ValueError(f"configuration fields must be positive: {', '.join(invalid)}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("attention head_dim must be even for RoPE")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.attention_type == "mha" and self.n_heads != self.n_kv_heads:
            raise ValueError("mha requires n_kv_heads == n_heads")
        dense_kv_width = 2 * self.n_kv_heads * self.head_dim
        if self.attention_type == "mla_lite" and self.mla_latent_dim >= dense_kv_width:
            raise ValueError("mla_latent_dim must be smaller than the dense K/V width")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.experts_per_token > self.n_experts:
            raise ValueError("experts_per_token cannot exceed n_experts")
        if self.moe_capacity_factor <= 0:
            raise ValueError("moe_capacity_factor must be positive")
        if self.moe_load_balance_coef < 0 or self.moe_router_z_loss_coef < 0:
            raise ValueError("MoE auxiliary-loss coefficients cannot be negative")
        if self.rope_base <= 1:
            raise ValueError("rope_base must be greater than 1")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
        if self.initialization_std <= 0:
            raise ValueError("initialization_std must be positive")
        if self.recurrent_halt_mode == "fixed" and self.recurrent_halt_threshold is not None:
            raise ValueError("recurrent_halt_threshold is only valid for drift/learned halting")
        if self.recurrent_halt_mode != "fixed":
            if self.recurrent_halt_threshold is None:
                raise ValueError("drift/learned halting requires recurrent_halt_threshold")
            if self.recurrent_halt_threshold <= 0:
                raise ValueError("recurrent_halt_threshold must be positive")
        if (
            self.recurrent_halt_mode == "learned"
            and self.recurrent_halt_threshold is not None
            and not 0 < self.recurrent_halt_threshold < 1
        ):
            raise ValueError("learned halt threshold must be in (0, 1)")
        if self.recurrent_min_steps > self.recurrent_depth:
            raise ValueError("recurrent_min_steps cannot exceed recurrent_depth")
        if self.recurrent_ponder_loss_coef < 0:
            raise ValueError("recurrent_ponder_loss_coef cannot be negative")
        if self.latent_scratchpad_size < 0:
            raise ValueError("latent_scratchpad_size cannot be negative")
        if self.dtype not in {"float32", "fp32", "float16", "fp16", "bfloat16", "bf16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if not self.device.strip():
            raise ValueError("device cannot be empty")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasonerConfig:
        if not isinstance(data, dict):
            raise TypeError("model config must be a mapping")
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown model config fields: {', '.join(unknown)}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReasonerConfig:
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("model config must be a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str = "default"
    seeds: list[int] = field(default_factory=lambda: [1337])
    seq_lens: list[int] = field(default_factory=lambda: [32, 64, 128])
    batch_size: int = 1
    timeout_s: float = 10.0
    output_dir: str = "reports"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("benchmark name cannot be empty")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("benchmark seeds must be unique")
        if not self.seq_lens or any(length < 1 for length in self.seq_lens):
            raise ValueError("all sequence lengths must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self.output_dir.strip():
            raise ValueError("output_dir cannot be empty")

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("benchmark config must be a mapping")
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown benchmark config fields: {', '.join(unknown)}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
