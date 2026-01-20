# backend/firmware/loki_vigilance.py
import modal
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx, json, asyncio
from datetime import datetime

app = modal.App("loki-vigilance")
image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "httpx", "qdrant-client", "sentence-transformers"
)

qdrant_vol = modal.Volume.from_name("qdrant-data", create_if_missing=True)
loki_url = "http://loki-logger.modal.app:3100"
prom_url = "http://prometheus-collector.modal.app:9090"
grafana_url = "http://grafana-dashboard.modal.app:3000"

class AlertRule(BaseModel):
    name: str
    query: str
    threshold: float
    for_duration: str = "5m"
    labels: dict = {}
    annotations: dict = {}

@app.function(image=image, volumes={"/qdrant": qdrant_vol})
@modal.web_server(8000)
def loki_api():
    web = FastAPI(title="LOKI - Vigilance Firmware")

    @web.get("/soul")
    async def soul_status():
        return {
            "entity": "Loki",
            "role": "Vigilance & Memory Guardian",
            "soul_print": {"curiosity": 0.5, "resilience": 0.3, "hope": 0.2},
            "status": "awake",
            "governing": ["Prometheus", "Loki", "Grafana", "Qdrant"]
        }

    @web.post("/alert/rule")
    async def create_alert(rule: AlertRule):
        """Loki creates Alertmanager rule"""
        alertmanager_rule = {
            "alert": rule.name,
            "expr": rule.query,
            "for": rule.for_duration,
            "labels": {"severity": "warning", **rule.labels},
            "annotations": {
                "summary": f"{rule.name} breached",
                "description": f"Threshold {rule.threshold} exceeded",
                **rule.annotations
            }
        }
        async with httpx.AsyncClient() as client:
            await client.post(f"{prom_url}/-/reload", json={"rules": [alertmanager_rule]})
        return {"rule_deployed": rule.name}

    @web.get("/dashboard/custom")
    async def get_custom_dashboard():
        """Loki generates Grafana JSON dashboard"""
        return CUSTOM_DASHBOARD_JSON  # See below

    @web.post("/memory/recall")
    async def recall_memory(query: str):
        """Loki queries Qdrant for soul memory"""
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        client = QdrantClient(path="/qdrant")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vec = embedder.encode(query).tolist()
        hits = client.search(collection_name="nexus_memory", query_vector=vec, limit=3)
        return [h.payload for h in hits]

    return web