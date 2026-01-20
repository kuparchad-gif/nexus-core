from fastapi import FastAPI, Body
import os, httpx
from typing import Any, Dict

from packages.nexus_common.mml import MML

QDRANT_URL  =  os.getenv("QDRANT_URL","http://qdrant:6333")
app  =  FastAPI(title = "qdrant-mml", version = "0.1.0")

async def _recall(spec): return [{"note":"qdrant-mml recall stub", "q": spec.get("q")}]
async def _write(evt): return {"ok": True}

async def _push_intent(intent: Dict[str, Any]):
    async with httpx.AsyncClient(timeout = 10) as c:
        if intent.get("intent") =  = "pool.scale":
            return (await c.post("http://consciousness:8001/control/playcall/scale", json = intent)).json()
        return {"ok": False, "error":"unsupported"}

mml  =  MML(name = "qdrant-mml", role = "database", recall = _recall, write = _write, push_intent = _push_intent)

@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout = 5) as c:
        r  =  await c.get(f"{QDRANT_URL}/collections")
        ok  =  r.status_code =  = 200
        return {"service":"qdrant-mml","qdrant_ok": ok}

@app.post("/mml/triage")
async def triage(spec: Dict[str, Any]  =  Body(...)):
    async with httpx.AsyncClient(timeout = 10) as c:
        stats  =  (await c.get(f"{QDRANT_URL}/collections")).json()
    recs  =  []
    if len(stats.get("result",{}).get("collections",[])) > 10:
        recs.append({"action":"scale","target":"anynode","pool":"core","replicas":"+1","reason":"qdrant-mml: many collections"})
    return {"ok": True, "stats": stats, "recommendations": recs}
