import torch
from wai_r0.config import ReasonerConfig
from wai_r0.model import CausalSelfAttention, MLALiteAttention, TopKMoE, RecurrentRefinement, ReasonerCore

def test_core_shape_and_generation():
    cfg=ReasonerConfig(vocab_size=32,d_model=32,d_ff=64,n_heads=4,n_kv_heads=4,n_layers=1,max_seq_len=16)
    core=ReasonerCore(cfg); x=torch.randint(0,cfg.vocab_size,(2,8)); y=core(x)
    assert list(y.shape)==[2,8,cfg.vocab_size]
    assert torch.isfinite(y).all()
    assert list(core.transformer.generate(torch.tensor([[1,2]]),2).shape)==[1,4]

def test_attention_stats():
    cfg=ReasonerConfig(d_model=32,d_ff=64,n_heads=4,n_kv_heads=2,attention_type='gqa')
    a=CausalSelfAttention(cfg); y=a(torch.randn(1,5,32)); assert list(y.shape)==[1,5,32]; assert a.last_stats.kv_cache_bytes>0

def test_mla_lite_compresses():
    cfg=ReasonerConfig(d_model=32,d_ff=64,n_heads=4,n_kv_heads=2,attention_type='mla_lite',mla_latent_dim=8)
    a=MLALiteAttention(cfg); y=a(torch.randn(1,5,32)); assert list(y.shape)==[1,5,32]; assert a.last_stats.compression_ratio<1

def test_moe_and_recurrent_stats():
    cfg=ReasonerConfig(d_model=16,d_ff=32,n_heads=4,n_kv_heads=4,use_moe=True,recurrent_depth=3)
    moe=TopKMoE(cfg); assert list(moe(torch.randn(2,4,16)).shape)==[2,4,16]; assert abs(sum(moe.last_stats.load_fraction)-1)<1e-6
    rec=RecurrentRefinement(cfg); assert list(rec(torch.randn(2,4,16)).shape)==[2,4,16]; assert rec.last_stats.depth==3
