from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, time, httpx

app = FastAPI(title="WireCrawler", version="0.1.0")

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8500")
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:9011")
JWT = os.getenv("NEXUS_JWT_SECRET", "change-me")

def log_loki(topic: str, payload: Dict[str, Any]):
    try:
        with httpx.Client(timeout=5.0) as s:
            s.post(f"{LOKI_URL}/log", json={"topic": topic, "payload": payload, "ts": time.time()})
    except Exception:
        pass

class WirePlan(BaseModel):
    require_caps: List[str] = Field(default_factory=lambda: ["router","relay"])
    # Direct wiring hints (cap -> urls)
    overrides: Dict[str, List[str]] = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"status": "ok", "registry": REGISTRY_URL}

@app.get("/scan_services")
def scan_services():
    with httpx.Client(timeout=10.0) as s:
        r = s.get(f"{REGISTRY_URL}/services")
        data = r.json()
        return data

@app.post("/wire")
def wire(plan: WirePlan):
    try:
        with httpx.Client(timeout=10.0) as s:
            r = s.get(f"{REGISTRY_URL}/services")
            services = r.json().get("services", {})
    except Exception as e:
        return {"error": str(e)}

    caps_map: Dict[str, List[str]] = {}
    for name, sv in services.items():
        for cap in sv.get("caps", []):
            caps_map.setdefault(cap, []).append(sv.get("url") or sv.get("public_url") or "")

    for cap, urls in plan.overrides.items():
        caps_map[cap] = list(set((caps_map.get(cap, []) or []) + urls))

    missing = [c for c in plan.require_caps if not caps_map.get(c)]
    result = {"caps": caps_map, "missing": missing}
    log_loki("wire.plan", result)
    return result
