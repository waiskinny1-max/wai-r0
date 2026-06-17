from wai_r0.config import ReasonerConfig
from wai_r0.benchmarks import zero_neural, memory, symbolic

def test_zero_and_memory_reports():
    cfg=ReasonerConfig(vocab_size=32,d_model=32,d_ff=64,n_heads=4,n_kv_heads=4,n_layers=1,max_seq_len=16,mla_latent_dim=8)
    z=zero_neural(cfg,batch_size=1,seq_len=8); assert z.result_type=='zero-training neural diagnostic'; assert z.raw_metrics['finite_logits']
    m=memory(cfg,'mha','mla_lite',[8,16]); assert len(m.raw_metrics['rows'])==2

def test_symbolic_report():
    r=symbolic('examples/tasks',budget_s=3,max_depth=1); assert r.result_type=='zero-training symbolic solver result'; assert r.raw_metrics['pass_at_1']>0
