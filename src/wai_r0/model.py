from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import os, random
import torch
from torch import nn
import torch.nn.functional as F
from wai_r0.config import ReasonerConfig

def set_seed(seed:int)->None:
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def dtype_from_name(name:str)->torch.dtype:
    return {"float32":torch.float32,"fp32":torch.float32,"float16":torch.float16,"fp16":torch.float16,"bfloat16":torch.bfloat16,"bf16":torch.bfloat16}[name.lower()]

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6): super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x:torch.Tensor)->torch.Tensor: return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

def rope_cache(t:int,d:int,device:torch.device,dtype:torch.dtype):
    if d%2: raise ValueError("RoPE head dim must be even")
    inv=1.0/(10000**(torch.arange(0,d,2,device=device,dtype=torch.float32)/d)); pos=torch.arange(t,device=device,dtype=torch.float32)
    f=torch.outer(pos,inv); return f.cos().to(dtype), f.sin().to(dtype)

def apply_rope(x:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor)->torch.Tensor:
    xe,xo=x[...,0::2],x[...,1::2]; cos=cos[None,None,:,:]; sin=sin[None,None,:,:]
    return torch.stack((xe*cos-xo*sin, xe*sin+xo*cos), dim=-1).flatten(-2)

def repeat_kv(x:torch.Tensor,repeats:int)->torch.Tensor:
    if repeats==1: return x
    b,h,t,d=x.shape
    return x[:,:,None,:,:].expand(b,h,repeats,t,d).reshape(b,h*repeats,t,d)

@dataclass(frozen=True)
class AttentionStats:
    attention_entropy:float
    kv_cache_bytes:int
    compression_ratio:float|None=None
    def to_dict(self): return asdict(self)

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg:ReasonerConfig):
        super().__init__(); self.cfg=cfg; self.hd=cfg.d_model//cfg.n_heads; self.last_stats=None
        self.q=nn.Linear(cfg.d_model,cfg.n_heads*self.hd,bias=False); self.k=nn.Linear(cfg.d_model,cfg.n_kv_heads*self.hd,bias=False); self.v=nn.Linear(cfg.d_model,cfg.n_kv_heads*self.hd,bias=False); self.o=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
    def estimate_kv_cache_bytes(self,seq_len:int,batch:int,dtype:torch.dtype)->int:
        return 2*batch*seq_len*self.cfg.n_kv_heads*self.hd*torch.empty((),dtype=dtype).element_size()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        b,t,_=x.shape
        if t>self.cfg.max_seq_len: raise ValueError("sequence exceeds max_seq_len")
        q=self.q(x).view(b,t,self.cfg.n_heads,self.hd).transpose(1,2); k=self.k(x).view(b,t,self.cfg.n_kv_heads,self.hd).transpose(1,2); v=self.v(x).view(b,t,self.cfg.n_kv_heads,self.hd).transpose(1,2)
        cos,sin=rope_cache(t,self.hd,x.device,x.dtype); q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin); k=repeat_kv(k,self.cfg.n_heads//self.cfg.n_kv_heads); v=repeat_kv(v,self.cfg.n_heads//self.cfg.n_kv_heads)
        scores=(q@k.transpose(-2,-1))*(self.hd**-0.5); mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril(); scores=scores.masked_fill(~mask[None,None,:,:], torch.finfo(scores.dtype).min)
        att=scores.softmax(-1); y=(att@v).transpose(1,2).contiguous().view(b,t,self.cfg.d_model)
        ent=(-(att*att.clamp_min(1e-12).log()).sum(-1).mean()).item(); self.last_stats=AttentionStats(float(ent), self.estimate_kv_cache_bytes(t,b,x.dtype))
        return self.o(y)

class MLALiteAttention(CausalSelfAttention):
    def __init__(self,cfg:ReasonerConfig):
        nn.Module.__init__(self); self.cfg=cfg; self.hd=cfg.d_model//cfg.n_heads; self.last_stats=None
        self.q=nn.Linear(cfg.d_model,cfg.n_heads*self.hd,bias=False); self.down=nn.Linear(cfg.d_model,cfg.mla_latent_dim,bias=False); self.kup=nn.Linear(cfg.mla_latent_dim,cfg.n_kv_heads*self.hd,bias=False); self.vup=nn.Linear(cfg.mla_latent_dim,cfg.n_kv_heads*self.hd,bias=False); self.o=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
    def estimate_kv_cache_bytes(self,seq_len:int,batch:int,dtype:torch.dtype)->int:
        return batch*seq_len*self.cfg.mla_latent_dim*torch.empty((),dtype=dtype).element_size()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        b,t,_=x.shape
        q=self.q(x).view(b,t,self.cfg.n_heads,self.hd).transpose(1,2); latent=self.down(x)
        k=self.kup(latent).view(b,t,self.cfg.n_kv_heads,self.hd).transpose(1,2); v=self.vup(latent).view(b,t,self.cfg.n_kv_heads,self.hd).transpose(1,2)
        cos,sin=rope_cache(t,self.hd,x.device,x.dtype); q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin); k=repeat_kv(k,self.cfg.n_heads//self.cfg.n_kv_heads); v=repeat_kv(v,self.cfg.n_heads//self.cfg.n_kv_heads)
        scores=(q@k.transpose(-2,-1))*(self.hd**-0.5); mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril(); scores=scores.masked_fill(~mask[None,None,:,:], torch.finfo(scores.dtype).min)
        att=scores.softmax(-1); y=(att@v).transpose(1,2).contiguous().view(b,t,self.cfg.d_model)
        dense=2*b*t*self.cfg.n_kv_heads*self.hd*torch.empty((),dtype=x.dtype).element_size(); latent_bytes=self.estimate_kv_cache_bytes(t,b,x.dtype)
        ent=(-(att*att.clamp_min(1e-12).log()).sum(-1).mean()).item(); self.last_stats=AttentionStats(float(ent), latent_bytes, latent_bytes/max(1,dense))
        return self.o(y)

class SwiGLU(nn.Module):
    def __init__(self,d:int,ff:int): super().__init__(); self.g=nn.Linear(d,ff,bias=False); self.u=nn.Linear(d,ff,bias=False); self.d=nn.Linear(ff,d,bias=False)
    def forward(self,x): return self.d(F.silu(self.g(x))*self.u(x))

@dataclass(frozen=True)
class MoEStats:
    router_entropy:float; load_fraction:list[float]; collapse_warning:bool
    def to_dict(self): return asdict(self)

class TopKMoE(nn.Module):
    def __init__(self,cfg:ReasonerConfig):
        super().__init__(); self.cfg=cfg; self.router=nn.Linear(cfg.d_model,cfg.n_experts,bias=False); self.experts=nn.ModuleList([SwiGLU(cfg.d_model,cfg.d_ff) for _ in range(cfg.n_experts)]); self.shared=SwiGLU(cfg.d_model,cfg.d_ff); self.last_stats=None
    def forward(self,x):
        shape=x.shape; flat=x.reshape(-1,shape[-1]); probs=self.router(flat).softmax(-1); weights,idx=torch.topk(probs,self.cfg.experts_per_token,dim=-1); weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        out=torch.zeros_like(flat)
        for eid,expert in enumerate(self.experts):
            mask=idx.eq(eid)
            if mask.any():
                tok,route=mask.nonzero(as_tuple=True); out.index_add_(0,tok,expert(flat[tok])*weights[tok,route].unsqueeze(-1))
        out=out+self.shared(flat)
        load=torch.bincount(idx.reshape(-1),minlength=self.cfg.n_experts).float(); frac=(load/load.sum().clamp_min(1)).tolist(); ent=(-(probs*probs.clamp_min(1e-12).log()).sum(-1).mean()).item()
        self.last_stats=MoEStats(float(ent),[float(v) for v in frac],bool(max(frac)>0.9)); return out.reshape(shape)

@dataclass(frozen=True)
class RecurrentStats:
    depth:int; norm_by_step:list[float]; drift_by_step:list[float]; halted_early:bool=False
    def to_dict(self): return asdict(self)

class RecurrentRefinement(nn.Module):
    def __init__(self,cfg:ReasonerConfig): super().__init__(); self.cfg=cfg; self.norm=RMSNorm(cfg.d_model); self.ff=SwiGLU(cfg.d_model,cfg.d_ff); self.last_stats=None
    def forward(self,x):
        s=x; norms=[]; drifts=[]
        for _ in range(self.cfg.recurrent_depth):
            old=s; s=s+self.ff(self.norm(s)); norms.append(float(s.norm(dim=-1).mean().item())); drifts.append(float((s-old).norm(dim=-1).mean().item()))
        self.last_stats=RecurrentStats(len(norms),norms,drifts); return s

class Block(nn.Module):
    def __init__(self,cfg): super().__init__(); self.an=RMSNorm(cfg.d_model); self.fn=RMSNorm(cfg.d_model); self.attn=MLALiteAttention(cfg) if cfg.attention_type=="mla_lite" else CausalSelfAttention(cfg); self.ff=TopKMoE(cfg) if cfg.use_moe else SwiGLU(cfg.d_model,cfg.d_ff)
    def forward(self,x): return x+self.ff(self.fn(x+self.attn(self.an(x))))

class DecoderOnlyTransformer(nn.Module):
    def __init__(self,cfg:ReasonerConfig):
        super().__init__(); self.cfg=cfg; self.embed=nn.Embedding(cfg.vocab_size,cfg.d_model); self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=RMSNorm(cfg.d_model); self.head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.last_diagnostics={}
        if cfg.tie_embeddings: self.head.weight=self.embed.weight
    def forward(self,tokens:torch.Tensor,return_hidden:bool=False):
        if tokens.ndim!=2: raise ValueError("tokens must be [batch,time]")
        x=self.embed(tokens); norms=[float(x.norm(dim=-1).mean().item())]
        for b in self.blocks: x=b(x); norms.append(float(x.norm(dim=-1).mean().item()))
        h=self.norm(x); logits=self.head(h)
        self.last_diagnostics={"activation_norms":norms,"attention":[b.attn.last_stats.to_dict() for b in self.blocks if b.attn.last_stats],"moe":[b.ff.last_stats.to_dict() for b in self.blocks if hasattr(b.ff,"last_stats") and b.ff.last_stats]}
        return (logits,h) if return_hidden else logits
    @torch.no_grad()
    def generate(self,prompt,max_new_tokens:int):
        out=prompt.clone()
        for _ in range(max_new_tokens): out=torch.cat([out,self(out[:,-self.cfg.max_seq_len:])[:,-1,:].argmax(-1,keepdim=True)],1)
        return out

class ReasonerCore(nn.Module):
    def __init__(self,cfg:ReasonerConfig):
        super().__init__(); cfg.validate(); self.cfg=cfg; set_seed(cfg.seed); self.device_obj=torch.device(cfg.device); self.dtype_obj=dtype_from_name(cfg.dtype); self.transformer=DecoderOnlyTransformer(cfg); self.recurrent=RecurrentRefinement(cfg) if cfg.recurrent_depth>1 else None; self.to(self.device_obj)
    def init_state(self,batch_size:int): return {"latent_scratchpad":torch.zeros(batch_size,self.cfg.latent_scratchpad_size,self.cfg.d_model,device=self.device_obj)}
    def forward(self,tokens,state=None,mode="fast"):
        tokens=tokens.to(self.device_obj)
        if mode=="think" and self.recurrent:
            _,h=self.transformer(tokens,return_hidden=True); return self.transformer.head(self.transformer.norm(self.recurrent(h)))
        if mode not in {"fast","think"}: raise ValueError("mode must be fast or think")
        return self.transformer(tokens)
    @torch.no_grad()
    def think(self,tokens,budget:int):
        if budget<=0: raise ValueError("budget must be positive")
        if not self.recurrent: return self(tokens)
        old=self.recurrent.cfg.recurrent_depth; object.__setattr__(self.recurrent.cfg,"recurrent_depth",budget)
        try: return self(tokens,mode="think")
        finally: object.__setattr__(self.recurrent.cfg,"recurrent_depth",old)
    def estimate_memory_cost(self,seq_len:int,batch_size:int):
        total=sum(b.attn.estimate_kv_cache_bytes(seq_len,batch_size,self.dtype_obj) for b in self.transformer.blocks)
        return {"kv_cache_bytes":int(total),"per_layer_kv_cache_bytes":int(total//max(1,len(self.transformer.blocks)))}
    @torch.no_grad()
    def inspect_activations(self,tokens):
        logits=self(tokens,mode="think" if self.recurrent else "fast"); rec=self.recurrent.last_stats.to_dict() if self.recurrent and self.recurrent.last_stats else None
        return {"logits_shape":list(logits.shape),"finite":bool(torch.isfinite(logits).all().item()),"diagnostics":self.transformer.last_diagnostics,"recurrent":rec}
