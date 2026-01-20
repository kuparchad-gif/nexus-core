# service/archiver_qdrant.py
import os, json, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from storage.qdrant_store import QdrantStore
from bookworms.system_worm import collect_system_snapshot
from bookworms.process_worm import collect_process_graph
from bookworms.services_worm import collect_services
from bookworms.models_worm import collect_models
from bookworms.memory_worm import collect_memory_artifacts
from bookworms.cluster_worm import collect_cluster_nodes

QDRANT_URL = os.environ.get("QDRANT_URL")  # e.g., [REDACTED-URL]
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

app = FastAPI(title="Nexus Soul Archiver (Qdrant)", version="1.0")
store = QdrantStore(url=QDRANT_URL, host=QDRANT_HOST, port=QDRANT_PORT, schema_path="schemas.json")

class CrawlRequest(BaseModel):
    which: list = ["system","process","services","models","memory","cluster"]

@app.post("/crawl")
def crawl(req: CrawlRequest):
    inserted = {}
    if "system" in req.which:
        inserted["systems"] = store.upsert_one("systems", collect_system_snapshot())
    if "process" in req.which:
        inserted["process_graph"] = store.upsert_many("process_graph", collect_process_graph(labels={"role":"archiver"}))
    if "services" in req.which:
        inserted["services"] = store.upsert_many("services", collect_services())
    if "models" in req.which:
        inserted["models"] = store.upsert_many("models", collect_models())
    if "memory" in req.which:
        inserted["memory_artifacts"] = store.upsert_many("memory_artifacts", collect_memory_artifacts())
    if "cluster" in req.which:
        inserted["cluster_nodes"] = store.upsert_many("cluster_nodes", collect_cluster_nodes())
    return {"status":"ok","inserted":inserted}

@app.get("/dump")
def dump():
    out = "dump"
    store.dump(out)
    return {"status":"ok","dump_dir": out}

@app.get("/health")
def health():
    return {"status":"ok","qdrant": QDRANT_URL or f"{QDRANT_HOST}:{QDRANT_PORT}"}

