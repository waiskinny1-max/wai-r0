from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml

AttentionType = Literal["mha", "gqa", "mla_lite"]

@dataclass(frozen=True)
class ReasonerConfig:
    vocab_size:int=64
    d_model:int=32
    n_layers:int=1
    n_heads:int=4
    n_kv_heads:int=4
    d_ff:int=64
    max_seq_len:int=64
    attention_type:AttentionType="mha"
    use_moe:bool=False
    n_experts:int=4
    experts_per_token:int=1
    recurrent_depth:int=1
    latent_scratchpad_size:int=8
    mla_latent_dim:int=12
    dropout:float=0.0
    tie_embeddings:bool=True
    dtype:str="float32"
    device:str="cpu"
    seed:int=1337
    def validate(self)->None:
        if self.d_model % self.n_heads: raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads: raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.attention_type=="mha" and self.n_heads!=self.n_kv_heads: raise ValueError("mha requires n_kv_heads == n_heads")
        if self.recurrent_depth<1: raise ValueError("recurrent_depth must be positive")
        if self.experts_per_token<1 or self.experts_per_token>self.n_experts: raise ValueError("experts_per_token out of range")
    @classmethod
    def from_dict(cls, data:dict[str,Any])->"ReasonerConfig":
        cfg=cls(**data); cfg.validate(); return cfg
    @classmethod
    def from_yaml(cls,path:str|Path)->"ReasonerConfig":
        raw=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw,dict): raise ValueError("model config must be mapping")
        return cls.from_dict(raw)
    def to_dict(self)->dict[str,Any]: return asdict(self)

@dataclass(frozen=True)
class BenchmarkConfig:
    name:str="default"
    seeds:list[int]=field(default_factory=lambda:[1337])
    seq_lens:list[int]=field(default_factory=lambda:[32,64,128])
    batch_size:int=1
    timeout_s:float=10.0
    output_dir:str="reports"
    @classmethod
    def from_yaml(cls,path:str|Path)->"BenchmarkConfig":
        raw=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**raw)
    def to_dict(self)->dict[str,Any]: return asdict(self)
