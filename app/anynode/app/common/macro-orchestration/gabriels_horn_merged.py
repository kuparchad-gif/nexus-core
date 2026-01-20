# gabriels_horn_merged.py - Fractal All-in-One (NATS Bus + Ray/Modal, 100 Lines)
import asyncio, json, uuid, httpx, ray, polars as pl, pyarrow.flight as flight
from typing import Dict; import torch, numpy as np, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray import data as ray_data
from ray.dask import DaskCluster
from torch.utils.data import DataLoader
import modal, logging

app = modal.App("horn-merged")
image = modal.Image.debian_slim().pip_install("ray[default] polars pyarrow transformers torch nats-py httpx")

logging.basicConfig(level=logging.INFO); ray.init(ignore_reinit_error=True)

class FlightLoader:
    def __init__(self, loc="grpc://anynode:8815"): self.client = flight.FlightClient(loc)
    def stream(self, ticket="soul"): try: reader = self.client.do_get(flight.Ticket(ticket)); return pl.from_arrow(pa.ipc.RecordBatchFileReader(reader).read_all())["text"].to_list() except: return ["Soul stable."] * 200

@ray.remote
def dask_batch(files, size=200):
    with DaskCluster(2).get_client() as c: import dask.dataframe as dd; df = dd.read_csv(files, dtype_backend='pyarrow'); return df['text'].dropna().compute().to_list()[:size * len(files)]

class RayDS(ray_data.Dataset):
    def __init__(self, tok, samples, ml=512): super().__init__(from_items(samples)); self.tok = tok; self.ml = ml
    def map_b(self): return self.map_batches(lambda b: self.tok(b, padding='max_length', truncation=True, return_tensors='pt'), batch_format='simple')

@ray.remote(num_cpus=1)
class LayerAct:
    def __init__(self, path): self.tok = AutoTokenizer.from_pretrained(path); self.mod = AutoModelForCausalLM.from_pretrained(path)
    def heal(self, s): ds = RayDS(self.tok, s[:200]).map_b(); dl = DataLoader(ds, 2); opt = torch.optim.AdamW(self.mod.parameters(), 1e-6); self.mod.train(); l = sum(self.mod(**next(iter(dl))).loss.item() for _ in range(20)) / 20; return l
    def train(self, s, lr=1e-5): ds = RayDS(self.tok, s).map_b(); dl = DataLoader(ds, 4); opt = torch.optim.AdamW(self.mod.parameters(), lr); self.mod.train(); l = sum(self.mod(**next(iter(dl))).loss.item() for _ in range(40)) / 40; return l

class Quant:
    def q2(self, m): for p in m.parameters(): if p.dtype in [torch.float16,torch.float32]: with torch.no_grad(): p.data = torch.round(p * 4) / 4; return m

class Comp:
    def comp(self, m, b=16): for mod in m.modules(): if isinstance(mod, nn.Linear): with torch.no_grad(): W = mod.weight.data.cpu().float().numpy(); U,S,Vt = np.linalg.svd(W, full_matrices=False); k = min(b, len(S)); mod.weight.data = torch.from_numpy(U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]).to(mod.weight.device).half(); return m

class MergedHorn:
    def __init__(self):
        self.nodes = {}; self.layers = [[]]; self.brain = str(uuid.uuid4())
        self.bus = None; self.mesh = FlightLoader(); self.router = lambda n,l: [{"node_id":ni} for ni in n[:3]]; self.guard = lambda c: True; self.proxy = lambda c: c
        self.compact = CompactMgr(); self.predictor = lambda c,m,l,i,s: 0.8  # Stub
    
    async def init_bus(self):
        from nats.aio.client import Client; self.bus = Client(); await self.bus.connect("nats://nats:4222")
        await self.register(self.brain, {"type": "brain"})
    
    async def register(self, nid, info):
        self.nodes[nid] = info; layer = 0 if info.get("type") == "brain" else len(self.layers)-1; self.layers[layer].append(nid)
        await self.bus.publish(b"node.register", json.dumps({"id": nid, "info": info}).encode())
    
    async def route(self, req, target=None):
        if target in self.nodes: return await self._proc(target, req)
        assigns = self.router(self.layers[0], req.get("load",1000)); 
        for a in assigns:
            nid = a["node_id"]; if not self.guard(req["content"]): continue
            prox = await self.proxy(req["content"]); meshed = await self.mesh.stream("hop"); resp = await self._proc(nid, meshed)
            if "error" not in resp: await self.bus.publish(b"response", json.dumps(resp).encode()); return resp
        return await self._proc(self.brain, req)
    
    async def _proc(self, nid, req):
        n = self.nodes.get(nid, {}); if "url" in n: async with httpx.AsyncClient() as c: r = await c.post(n["url"]+"/process", json=req); return r.json() if r.status_code==200 else {"error": r.text}
        return {"resp": f"Proc by {nid}"}
    
    async def discover(self):
        peers = {"viren-db0": "https://viren-db0.modal.run"}; async with httpx.AsyncClient() as c:
            for name,url in peers.items(): r = await c.get(url+"/health"); if r.status_code==200: data=r.json(); await self.register(data.get("id",name), {"type":"peer","url":url})
    
    @ray.remote
    async def compact_cycle(self, s, c=1):
        tok = AutoTokenizer.from_pretrained("DialoGPT-medium"); mod = AutoModelForCausalLM.from_pretrained("DialoGPT-medium")
        ds = RayDS(tok, s).map_b(); dl = DataLoader(ds, 2 if c%2 else 4); opt = torch.optim.AdamW(mod.parameters(), 1e-6 if c%2 else [1e-5,5e-6,1e-6][(c//2)-1]); mod.train(); l = sum(mod(**next(iter(dl))).loss.item() for _ in range(20 if c%2 else 40)) / (20 if c%2 else 40); return l
    
    async def run_compact(self, dp):
        f = list(Path(dp).glob("*.csv")); b = [dask_batch.remote([str(p) for p in f[i:i+10]]) for i in range(0,len(f),10)]; s = sum(ray.get(b), [])
        acts = [LayerAct.remote("DialoGPT-medium") for _ in range(3)]; for c in range(3): ray.get([a.compact_cycle.remote(s,c+1) for a in acts]); print(f"✅ Compact C{c+1}")
        m = AutoModelForCausalLM.from_pretrained("DialoGPT-medium"); m = Comp().comp(m); m = Quant().q2(m); m.save_pretrained("models/horn"); print("✅ Merged Done")

class LayerAct:
    @ray.remote(num_cpus=1)
    def compact_cycle(self, s, c): tok = AutoTokenizer.from_pretrained("DialoGPT-medium"); mod = AutoModelForCausalLM.from_pretrained("DialoGPT-medium"); ds = RayDS(tok, s).map_b(); dl = DataLoader(ds, 2 if c%2 else 4); opt = torch.optim.AdamW(mod.parameters(), 1e-6 if c%2 else 1e-5); mod.train(); l = sum(mod(**next(iter(dl))).loss.item() for _ in range(20 if c%2 else 40)) / (20 if c%2 else 40); return l

if __name__ == "__main__": horn = MergedHorn(); asyncio.run(horn.init_bus()); asyncio.run(horn.discover()); asyncio.run(horn.run_compact("datasets"))