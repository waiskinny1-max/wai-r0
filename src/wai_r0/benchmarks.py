from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore, set_seed
from wai_r0.report import BenchmarkReport, recommend
from wai_r0.symbolic import load_task, ProgramSearch

def zero_neural(cfg:ReasonerConfig,batch_size=1,seq_len=16):
    set_seed(cfg.seed); core=ReasonerCore(cfg); tokens=torch.randint(0,cfg.vocab_size,(batch_size,min(seq_len,cfg.max_seq_len)),device=core.device_obj); logits=core(tokens,mode="think" if cfg.recurrent_depth>1 else "fast"); loss=logits.float().mean(); loss.backward(); grads=[p.grad.detach().float().norm().item() for p in core.parameters() if p.grad is not None]; finite=bool(torch.isfinite(logits).all().item()); fgrads=all(torch.isfinite(torch.tensor(g)).item() for g in grads); insp=core.inspect_activations(tokens); moe=insp["diagnostics"].get("moe",[]); collapsed=any(m.get("collapse_warning",False) for m in moe); rec=insp.get("recurrent"); score=.4*finite+.3*fgrads+.2*(not collapsed)+.1*(rec is None or max(rec.get("norm_by_step",[0]))<1e4)
    return BenchmarkReport("zero_neural","zero-training neural diagnostic",cfg.seed,cfg.device,cfg.dtype,cfg.to_dict(),raw_metrics={"finite_logits":finite,"finite_gradients":fgrads,"grad_norm_mean":sum(grads)/max(1,len(grads)),"inspection":insp,"r0_stability_score":score},summary="Random-weight numerical diagnostic completed. This is not an intelligence result.",limitations=["Random weights do not reason.","Single local run; vary seeds and tasks before scaling."],recommendation=recommend(score if finite and fgrads else .0))

def _attn_cfg(cfg,attn):
    n_kv=cfg.n_heads if attn=="mha" else (max(1,cfg.n_heads//2) if attn in {"gqa","mla_lite"} else cfg.n_kv_heads)
    return ReasonerConfig.from_dict({**cfg.to_dict(),"attention_type":attn,"n_kv_heads":n_kv})
def memory(cfg,baseline,candidate,seq_lens,batch_size=1):
    b=ReasonerCore(_attn_cfg(cfg,baseline)); c=ReasonerCore(_attn_cfg(cfg,candidate)); rows=[]
    for s in seq_lens:
        bb=b.estimate_memory_cost(s,batch_size)["kv_cache_bytes"]; cc=c.estimate_memory_cost(s,batch_size)["kv_cache_bytes"]; rows.append({"seq_len":s,"baseline_bytes":bb,"candidate_bytes":cc,"candidate_over_baseline":cc/max(1,bb),"saved_bytes":bb-cc})
    ratio=sum(r["candidate_over_baseline"] for r in rows)/len(rows)
    return BenchmarkReport("memory","architecture-prior diagnostic",cfg.seed,cfg.device,cfg.dtype,cfg.to_dict(),{"baseline":baseline,"candidate":candidate,"seq_lens":seq_lens}, {"rows":rows,"average_candidate_over_baseline":ratio},"KV-cache memory estimate. Not a reasoning benchmark.",["Static estimate; profile target hardware.","MLA-lite is not DeepSeek MLA."], "TINY-TRAIN ONLY — promising but unproven." if ratio<1 else "DO NOT TRAIN — architecture has no useful signal.")

def symbolic(tasks,budget_s=10.0,max_depth=2):
    paths=sorted(Path(tasks).glob("*.json"));
    if not paths: raise FileNotFoundError(f"no .json tasks found in {tasks}")
    search=ProgramSearch(max_depth=max_depth); results=[]; solved=0
    for p in paths:
        task=load_task(p); r=search.solve(task,budget_s); solved+=int(r.solved); results.append({"task_id":task.task_id,**r.to_dict()})
    pass1=solved/len(paths)
    return BenchmarkReport("symbolic_arc","zero-training symbolic solver result",0,"cpu","n/a",benchmark_config={"tasks":str(tasks),"budget_s":budget_s,"max_depth":max_depth},raw_metrics={"tasks":results,"pass_at_1":pass1},summary="Explicit symbolic program search. Not neural reasoning.",limitations=["Small DSL and demo tasks.","Avoid public-eval leakage."],recommendation="TINY-TRAIN ONLY — promising but unproven." if pass1>0 else "DO NOT TRAIN — architecture has no useful signal.")

def tiny_train(cfg,task="copy",examples=8,batch_size=4,seq_len=8):
    core=ReasonerCore(cfg); opt=torch.optim.AdamW(core.parameters(),lr=3e-4); initial=None; final=None
    for _ in range(max(1,examples//batch_size)):
        x=torch.randint(1,cfg.vocab_size,(batch_size,min(seq_len,cfg.max_seq_len)),device=core.device_obj); y=x.clone() if task=="copy" else torch.flip(x,[1]); opt.zero_grad(set_to_none=True); logits=core(x,mode="think" if cfg.recurrent_depth>1 else "fast"); loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1)); initial=float(loss.item()) if initial is None else initial; loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(),1.0); opt.step(); final=float(loss.item())
    delta=float(initial-final)
    return BenchmarkReport("tiny_train","tiny-training architecture probe",cfg.seed,cfg.device,cfg.dtype,cfg.to_dict(),{"task":task,"examples":examples}, {"task":task,"examples":examples,"initial_loss":initial,"final_loss":final,"loss_delta":delta},"Tiny supervised algorithmic probe. Not pretrained reasoning.",["Tiny smoke budget only.","Run multiple seeds and OOD lengths for real decisions."],"TINY-TRAIN ONLY — promising but unproven." if delta>0 else "DO NOT TRAIN — architecture has no useful signal.")

def ablate(cfg,matrix_path):
    import yaml
    raw=yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8")); rows=[]
    for v in raw["variants"]:
        attn=v["attention"]; n_kv=cfg.n_heads if attn=="mha" else max(1,cfg.n_heads//2); vc=ReasonerConfig.from_dict({**cfg.to_dict(),"attention_type":attn,"n_kv_heads":n_kv,"use_moe":bool(v.get("moe",False)),"recurrent_depth":2 if v.get("recurrent",False) else 1}); r=zero_neural(vc,batch_size=1,seq_len=8); rows.append({"variant":v["name"],"metrics":r.raw_metrics,"recommendation":r.recommendation})
    stable=sum(row["metrics"].get("finite_logits") and row["metrics"].get("finite_gradients") for row in rows)
    return BenchmarkReport("ablation","mixed architecture diagnostic",cfg.seed,cfg.device,cfg.dtype,cfg.to_dict(),{"matrix":str(matrix_path)}, {"variants":rows,"stable_count":stable,"total":len(rows)},"Ablation over attention, MoE, and recurrence. Diagnostic only.",["Uses zero-neural suite only in v0.1.","Add memory and tiny-training subruns before scale decisions."],"TINY-TRAIN ONLY — promising but unproven." if stable else "DO NOT TRAIN — architecture has no useful signal.")
