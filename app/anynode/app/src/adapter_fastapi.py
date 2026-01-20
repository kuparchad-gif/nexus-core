import os
from fastapi import FastAPI, Request, HTTPException

try:
    from nexus_core.stem.envelopes import Envelope, Response
except Exception:
    import sys, pathlib
    sys.path.append(str(pathlib.Path('.').resolve()))
    from nexus_core.stem.envelopes import Envelope, Response

from ...stem-adapters.common.loki_client import push_log
from ...stem-adapters.common.memory_io import memory_upsert, memory_query

app = FastAPI(title="CogniKube Memory Adapter", version="v1")
TOKEN = os.environ.get("CUBE_TOKEN")
DEFAULT_COLLECTION = os.environ.get("MEM_COLLECTION","nexus.mem")

@app.post("/write", response_model=Response)
async def write(env: Envelope, request: Request):
    if TOKEN and request.headers.get("x-cube-token") != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")
    coll = env.payload.get("collection", DEFAULT_COLLECTION)
    vec = env.payload.get("vector", [0.0, 0.0, 0.0])
    pay = env.payload.get("payload", {"text": env.payload.get("text","")})
    pt = {"id": env.signal_id, "vector": vec, "payload": pay}
    mode = memory_upsert(coll, [pt]).get("mode","local")
    push_log("INFO", f"mem.write coll={coll} mode={mode}", {"kind":"memory","triops":"Nexus TriOps"})
    return Response(signal_id=env.signal_id, ok=True, result={"mode": mode})

@app.get("/recent")
async def recent(collection: str = None, limit: int = 5):
    coll = collection or DEFAULT_COLLECTION
    out = memory_query(coll, limit=limit)
    return {"collection": coll, "items": out.get("points",[])}
