from __future__ import annotations
import argparse
from pathlib import Path
from wai_r0.config import ReasonerConfig
from wai_r0.benchmarks import zero_neural, memory, symbolic, tiny_train, ablate
from wai_r0.report import markdown_from_json

def seqs(s): return [int(x) for x in s.split(',') if x]
def duration(s):
    s=str(s).strip().lower(); mult=.001 if s.endswith('ms') else 1.0; s=s[:-2] if s.endswith('ms') else (s[:-1] if s.endswith('s') else s); return float(s)*mult
def cfg(path): return ReasonerConfig.from_yaml(path or 'configs/model/nano.yaml')
def emit(report,out=None,base='latest'):
    outdir=Path(out).parent if out else Path('reports'); stem=Path(out).stem if out else base; jp,mp=report.write(outdir,stem); Path('reports').mkdir(exist_ok=True); Path('reports/latest.json').write_text(jp.read_text(),encoding='utf-8'); Path('reports/latest.md').write_text(mp.read_text(),encoding='utf-8'); print(jp); print(mp)
def main(argv=None):
    p=argparse.ArgumentParser(prog='wai-r0'); sub=p.add_subparsers(dest='cmd',required=True)
    z=sub.add_parser('zero-neural'); z.add_argument('--config',default='configs/model/nano.yaml'); z.add_argument('--batch-size',type=int,default=1); z.add_argument('--seq-len',type=int,default=16); z.add_argument('--output')
    m=sub.add_parser('memory'); m.add_argument('--config',default='configs/model/nano.yaml'); m.add_argument('--baseline',default='mha'); m.add_argument('--candidate',default='mla_lite'); m.add_argument('--seq-lens',type=seqs,required=True); m.add_argument('--batch-size',type=int,default=1); m.add_argument('--output')
    s=sub.add_parser('symbolic-arc'); s.add_argument('--tasks',required=True); s.add_argument('--budget',type=duration,default=10.0); s.add_argument('--max-depth',type=int,default=2); s.add_argument('--output')
    t=sub.add_parser('tiny-train'); t.add_argument('--config','--model',dest='config',default='configs/model/nano.yaml'); t.add_argument('--task',choices=['copy','reverse'],default='copy'); t.add_argument('--examples',type=int,default=8); t.add_argument('--output')
    a=sub.add_parser('ablate'); a.add_argument('--config',default='configs/model/nano.yaml'); a.add_argument('--matrix',required=True); a.add_argument('--output')
    r=sub.add_parser('report'); r.add_argument('--input',required=True); r.add_argument('--format',choices=['md'],default='md'); r.add_argument('--output')
    args=p.parse_args(argv)
    if args.cmd=='zero-neural': emit(zero_neural(cfg(args.config),args.batch_size,args.seq_len),args.output,'zero_neural')
    elif args.cmd=='memory': emit(memory(cfg(args.config),args.baseline,args.candidate,args.seq_lens,args.batch_size),args.output,'memory')
    elif args.cmd=='symbolic-arc': emit(symbolic(args.tasks,args.budget,args.max_depth),args.output,'symbolic_arc')
    elif args.cmd=='tiny-train': emit(tiny_train(cfg(args.config),args.task,args.examples),args.output,'tiny_train')
    elif args.cmd=='ablate': emit(ablate(cfg(args.config),args.matrix),args.output,'ablation')
    elif args.cmd=='report':
        md=markdown_from_json(args.input)
        if args.output: Path(args.output).write_text(md,encoding='utf-8'); print(args.output)
        else: print(md)
    return 0
