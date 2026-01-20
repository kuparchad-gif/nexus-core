from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os, time, httpx

app = FastAPI(title="StarterCrawler", version="0.1.0")

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8500")
STARTERS = os.getenv("SERVICE_STARTERS_JSON", "{}")  # e.g., {"router":"http://gw.starter/start","relay":"http://spine.starter/start"}
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:9011")

def log_loki(topic: str, payload: Dict[str, Any]):
    try:
        with httpx.Client(timeout=5.0) as s:
            s.post(f"{LOKI_URL}/log", json={"topic": topic, "payload": payload, "ts": time.time()})
    except Exception:
        pass

class StartPlan(BaseModel):
    probe_paths: Dict[str, str] = {}  # cap -> path (e.g., "/health")
    timeouts: Dict[str, float] = {}   # cap -> seconds

@app.get("/health")
def health():
    return {"status":"ok","registry":REGISTRY_URL}

@app.post("/start_missing")
def start_missing(plan: StartPlan):
    try:
        with httpx.Client(timeout=10.0) as s:
            reg = s.get(f"{REGISTRY_URL}/services").json().get("services", {})
    except Exception as e:
        return {"error":str(e)}

    starters = {}
    try:
        starters = json.loads(STARTERS)
    except Exception:
        starters = {}

    missing = []
    for name, sv in reg.items():
        for cap in sv.get("caps", []):
            url = sv.get("url") or sv.get("public_url")
            path = plan.probe_paths.get(cap, "/__cell__/health")
            try:
                with httpx.Client(timeout=plan.timeouts.get(cap, 2.0)) as s:
                    r = s.get((url or "") + path)
                    if r.status_code >= 400:
                        missing.append({"cap":cap,"name":name,"url":url,"status":r.status_code})
            except Exception:
                missing.append({"cap":cap,"name":name,"url":url,"status":"unreachable"})
    # Try to start each missing cap via starter webhook if present
    actions = []
    for m in missing:
        hook = starters.get(m["cap"])
        if hook:
            try:
                with httpx.Client(timeout=5.0) as s:
                    rr = s.post(hook, json=m)
                    actions.append({"target":m,"starter":hook,"result":rr.status_code})
            except Exception as e:
                actions.append({"target":m,"starter":hook,"error":str(e)})
    out = {"missing": missing, "actions": actions}
    log_loki("starter.actions", out)
    return out
