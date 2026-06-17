from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
import json, time
from pathlib import Path
from collections import deque

GridData=tuple[tuple[int,...],...]
@dataclass(frozen=True)
class Grid:
    data:GridData
    def __post_init__(self):
        if not self.data or not self.data[0]: raise ValueError("empty grid")
        if any(len(r)!=len(self.data[0]) for r in self.data): raise ValueError("ragged grid")
    @classmethod
    def from_lists(cls, rows): return cls(tuple(tuple(int(c) for c in r) for r in rows))
    @property
    def height(self): return len(self.data)
    @property
    def width(self): return len(self.data[0])
    def to_lists(self): return [list(r) for r in self.data]
    def colors(self): return {c for r in self.data for c in r}
    def map(self,fn): return Grid(tuple(tuple(fn(c) for c in r) for r in self.data))

def identity(g): return g
def rotate90(g): return Grid(tuple(tuple(row[i] for row in reversed(g.data)) for i in range(g.width)))
def rotate180(g): return Grid(tuple(tuple(reversed(r)) for r in reversed(g.data)))
def rotate270(g): return Grid(tuple(tuple(row[i] for row in g.data) for i in reversed(range(g.width))))
def mirror_x(g): return Grid(tuple(reversed(g.data)))
def mirror_y(g): return Grid(tuple(tuple(reversed(r)) for r in g.data))
def infer_background(g):
    counts={}
    for r in g.data:
        for c in r: counts[c]=counts.get(c,0)+1
    return max(counts.items(),key=lambda kv:kv[1])[0]
def crop_nonzero(g):
    bg=infer_background(g); pts=[(r,c) for r,row in enumerate(g.data) for c,v in enumerate(row) if v!=bg]
    if not pts: return g
    r0,r1=min(r for r,_ in pts),max(r for r,_ in pts); c0,c1=min(c for _,c in pts),max(c for _,c in pts)
    return Grid(tuple(tuple(g.data[r][c] for c in range(c0,c1+1)) for r in range(r0,r1+1)))
def pad(g,amount=1,color=None):
    bg=infer_background(g) if color is None else color; width=g.width+2*amount; border=tuple(bg for _ in range(width)); rows=[border]*amount+[tuple([bg]*amount+list(r)+[bg]*amount) for r in g.data]+[border]*amount; return Grid(tuple(rows))
def replace_color(g,old,new): return g.map(lambda c:new if c==old else c)
def recolor(g,mapping): return g.map(lambda c:mapping.get(c,c))
def extract_connected_components(g,background=None):
    bg=infer_background(g) if background is None else background; seen=set(); comps=[]
    for r in range(g.height):
        for c in range(g.width):
            if (r,c) in seen or g.data[r][c]==bg: continue
            color=g.data[r][c]; q=deque([(r,c)]); seen.add((r,c)); comp=set()
            while q:
                cr,cc=q.popleft(); comp.add((cr,cc))
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc=cr+dr,cc+dc
                    if 0<=nr<g.height and 0<=nc<g.width and (nr,nc) not in seen and g.data[nr][nc]==color: seen.add((nr,nc)); q.append((nr,nc))
            comps.append(comp)
    return comps
def count_components(g): return len(extract_connected_components(g))
def bounding_box(g):
    bg=infer_background(g); pts=[(r,c) for r,row in enumerate(g.data) for c,v in enumerate(row) if v!=bg]
    return None if not pts else (min(r for r,_ in pts),min(c for _,c in pts),max(r for r,_ in pts),max(c for _,c in pts))
def translate_object(g,dr=0,dc=0):
    bg=infer_background(g); out=[[bg]*g.width for _ in range(g.height)]
    for r,row in enumerate(g.data):
        for c,v in enumerate(row):
            if v!=bg and 0<=r+dr<g.height and 0<=c+dc<g.width: out[r+dr][c+dc]=v
    return Grid.from_lists(out)
def scale_object(g,factor=2):
    rows=[]
    for row in g.data:
        exp=[c for c in row for _ in range(factor)]
        rows += [exp.copy() for _ in range(factor)]
    return Grid.from_lists(rows)
def tile_pattern(g,rows=2,cols=2): return Grid.from_lists([list(r)*cols for _ in range(rows) for r in g.data])
def extend_line(g):
    bg=infer_background(g); out=g.to_lists()
    for r,row in enumerate(g.data):
        for color in {v for v in row if v!=bg}:
            idx=[i for i,v in enumerate(row) if v==color]
            if len(idx)>=2:
                for c in range(min(idx),max(idx)+1): out[r][c]=color
    for c in range(g.width):
        col=[g.data[r][c] for r in range(g.height)]
        for color in {v for v in col if v!=bg}:
            idx=[i for i,v in enumerate(col) if v==color]
            if len(idx)>=2:
                for r in range(min(idx),max(idx)+1): out[r][c]=color
    return Grid.from_lists(out)
def fill_enclosed(g):
    bg=0 if 0 in g.colors() else infer_background(g); out=g.to_lists(); exterior=set(); q=deque()
    for r in range(g.height):
        for c in (0,g.width-1):
            if g.data[r][c]==bg: q.append((r,c)); exterior.add((r,c))
    for c in range(g.width):
        for r in (0,g.height-1):
            if g.data[r][c]==bg and (r,c) not in exterior: q.append((r,c)); exterior.add((r,c))
    while q:
        r,c=q.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<g.height and 0<=nc<g.width and (nr,nc) not in exterior and g.data[nr][nc]==bg: exterior.add((nr,nc)); q.append((nr,nc))
    fill=next((v for row in g.data for v in row if v!=bg),1)
    for r in range(g.height):
        for c in range(g.width):
            if g.data[r][c]==bg and (r,c) not in exterior: out[r][c]=fill
    return Grid.from_lists(out)
def object_union(a,b): return a if a==b else a
def object_intersection(a,b): return a if a==b else Grid.from_lists([[0]*a.width for _ in range(a.height)])
def object_difference(a,b): return Grid.from_lists([[0 if b.data[r][c]!=0 else a.data[r][c] for c in range(a.width)] for r in range(a.height)])

PRIMITIVES={"identity":identity,"rotate90":rotate90,"rotate180":rotate180,"rotate270":rotate270,"mirror_x":mirror_x,"mirror_y":mirror_y,"crop_nonzero":crop_nonzero,"pad":pad,"extend_line":extend_line,"fill_enclosed":fill_enclosed,"scale_object":scale_object,"tile_pattern":tile_pattern}
@dataclass(frozen=True)
class Program:
    names:tuple[str,...]
    def __call__(self,g):
        out=g
        for n in self.names: out=PRIMITIVES[n](out)
        return out
    def describe(self): return " -> ".join(self.names)
@dataclass(frozen=True)
class TaskExample: input:Grid; output:Grid|None=None
@dataclass(frozen=True)
class ArcTask: task_id:str; train:tuple[TaskExample,...]; test:tuple[TaskExample,...]
@dataclass(frozen=True)
class SearchResult:
    solved:bool; program:str|None; predictions:list; candidates_tested:int; elapsed_s:float; failure:str|None=None
    def to_dict(self): return asdict(self)
class ProgramSearch:
    def __init__(self,max_depth=2,beam_size=None): self.max_depth=max_depth; self.beam_size=beam_size
    def enumerate_programs(self):
        names=[n for n in PRIMITIVES if n!="identity"]; progs=[Program(("identity",))]
        for d in range(1,self.max_depth+1): progs += [Program(tuple(c)) for c in product(names, repeat=d)]
        return progs[:self.beam_size] if self.beam_size else progs
    def solve(self,task,timeout_s=10.0):
        start=time.perf_counter(); tested=0
        for p in self.enumerate_programs():
            if time.perf_counter()-start>timeout_s: return SearchResult(False,None,[],tested,time.perf_counter()-start,"timeout")
            tested+=1
            try: ok=all(ex.output is not None and p(ex.input)==ex.output for ex in task.train)
            except Exception: ok=False
            if ok: return SearchResult(True,p.describe(),[p(ex.input).to_lists() for ex in task.test],tested,time.perf_counter()-start)
        return SearchResult(False,None,[],tested,time.perf_counter()-start,"no verified program")
def load_task(path):
    raw=json.loads(Path(path).read_text(encoding="utf-8")); train=tuple(TaskExample(Grid.from_lists(e["input"]),Grid.from_lists(e["output"])) for e in raw["train"]); test=tuple(TaskExample(Grid.from_lists(e["input"]),Grid.from_lists(e["output"]) if "output" in e else None) for e in raw["test"]); return ArcTask(raw.get("id",Path(path).stem),train,test)
class ReasoningController:
    def __init__(self,max_depth=2): self.searcher=ProgramSearch(max_depth=max_depth)
    def solve(self,task,budget): return {"result_type":"zero-training symbolic solver result","symbolic":self.searcher.solve(task,budget).to_dict()}
