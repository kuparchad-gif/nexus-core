import os, random, hashlib, json
from typing import Any, Dict, List
from fastapi import APIRouter, Body
import httpx

from packages.nexus_common.mml import MML
from packages.nexus_common.qdrant_mem import ensure_collection, upsert_points

router = APIRouter(prefix="/mml", tags=["mml"])

COLL = "mml_consciousness"
EMB_DIM = int(os.getenv("MML_EMB_DIM","64"))

def _embed(text: str, dim: int=EMB_DIM) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    base = list(h) * ((dim // len(h)) + 1)
    return [x/255.0 for x in base[:dim]]

async def _write(evt: Dict[str, Any]):
    await ensure_collection(COLL, size=EMB_DIM, distance="Cosine")
    vec = _embed(evt.get("line") or json.dumps(evt, sort_keys=True))
    pt = {"id": random.randint(1, 2**31-1), "vector": vec, "payload": evt}
    return await upsert_points(COLL, [pt])

async def _recall(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"note":"recall stub for consciousness", "q": spec.get("q")}]

async def _push_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    reason = intent.get("reason","mml action")
    t = intent.get("intent")
    async with httpx.AsyncClient(timeout=10) as c:
        if t == "pool.scale":
            r = await c.post("[REDACTED-URL] json=intent)
        elif t == "route.update":
            r = await c.post("[REDACTED-URL] json={"routes": intent.get("changes", []), "reason": reason})
        elif t == "data.purge":
            r = await c.post("[REDACTED-URL] json=intent)
        else:
            return {"ok": False, "error":"unknown_intent"}
        return r.json()

mml = MML(name="consciousness-mml", role="consciousness", recall=_recall, write=_write, push_intent=_push_intent)

@router.post("/observe")
async def observe(evt: Dict[str, Any] = Body(...)):
    return await mml.observe(evt)

@router.post("/reason")
async def reason(spec: Dict[str, Any] = Body(...)):
    q = spec.get("q","status?")
    return await mml.reason(q, k=int(spec.get("k",20)))

@router.post("/act")
async def act(intent: Dict[str, Any] = Body(...)):
    return await mml.act(intent)

