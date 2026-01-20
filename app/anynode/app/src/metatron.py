import time, math, uuid, hashlib
from typing import List, Tuple, Dict

FACETS=['temporal','causal','agentic','modal','topical','spatial']

def now(): return time.time()
def _vec(txt:str):
  b=[0.0]*64
  for w in txt.lower().split():
    h=int(hashlib.sha256(w.encode()).hexdigest(),16); b[h%64]+=1.0
  n=math.sqrt(sum(x*x for x in b)) or 1.0
  return [x/n for x in b]
def _cos(a,b):
  num=sum(x*y for x,y in zip(a,b)); da=math.sqrt(sum(x*x for x in a)) or 1.0; db=math.sqrt(sum(y*y for y in b)) or 1.0
  return num/(da*db)

class Atom:
  def __init__(s,val,mod='text',tags=None): s.id=f'atom:{uuid.uuid4()}'; s.ts=now(); s.mod=mod; s.tags=tags or []; s.val=val; s.vec=_vec(val)

class Capsule:
  def __init__(s,atoms,syn,P,lg,tags=None): s.id=f'cap:{uuid.uuid4()}'; s.at=[a.id for a in atoms]; s.syn=syn; s.P=float(P); s.tags=tags or []; s.lg=lg; s.ts=now()

class Lattice:
  def __init__(s): s.atoms={}; s.caps={}; s.edges={f:{} for f in FACETS}
  def add_atom(s,a:Atom):
    s.atoms[a.id]=a
    for o in list(s.atoms.values())[-256:]:
      if o.id==a.id: continue
      dt=abs(a.ts-o.ts); w=max(0.0,1.0-dt/600.0)
      if w>0: s.edges['temporal'].setdefault(a.id,{})[o.id]=w; s.edges['temporal'].setdefault(o.id,{})[a.id]=w
  def link(s,f:str,src:str,dst:str,w:float=1.0): s.edges.setdefault(f,{}).setdefault(src,{})[dst]=w
  def promote(s,atoms,syn,P,lg,tags=None):
    c=Capsule(atoms,syn,P,lg,tags); s.caps[c.id]=c
    for a in atoms: s.link('modal',c.id,a.id,P); s.link('modal',a.id,c.id,P)
    return c
  def vec_search(s,qv:List[float],k:int=5)->List[Tuple[str,float]]:
    R=[(a.id,_cos(qv,a.vec)) for a in s.atoms.values()]; R.sort(key=lambda x:x[1],reverse=True); return R[:k]
  def geo(s,seeds,fw:Dict[str,float],k:int=5,beam:int=32,depth:int=3):
    seen={}; fr=[(sid,1.0,[(sid,'seed',1.0)]) for sid in seeds]
    for _ in range(depth):
      tmp=[]
      for nid,sc,path in fr:
        for f,wf in fw.items():
          for dst,w in s.edges.get(f,{}).get(nid,{}).items():
            ns=sc*(0.6+0.4*w*max(0.0,wf))
            if ns>seen.get(dst,0): seen[dst]=ns; tmp.append((dst,ns,path+[(dst,f,w)]))
      tmp.sort(key=lambda x:x[1],reverse=True); fr=tmp[:beam]
    out=[]
    for nid,sc,path in fr[:k]:
      fac={}
      for _,f,w in path:
        if f!='seed': fac[f]=fac.get(f,0.0)+w
      out.append((nid,sc,fac))
    return out

def perm(s,h,t,r,u,ws=0.35,wh=0.25,wt=0.25,wr=0.10,wu=0.05):
  clamp=lambda x:max(0.0,min(1.0,x)); s,h,t,r,u=map(clamp,(s,h,t,r,u)); tot=ws+wh+wt+wr+wu
  return (ws*s+wh*h+wt*t+wr*r+wu*u)/tot

def qvec(txt:str): return _vec(txt)
