from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os, time, docker, httpx, json

app = FastAPI(title="OrganizerBot", version="0.1.0")

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8500")
WIRE_URL     = os.getenv("WIRE_URL", "http://localhost:9021")
STARTER_URL  = os.getenv("STARTER_URL", "http://localhost:9022")
ARCHIVER_URL = os.getenv("ARCHIVER_URL", "http://localhost:9020")

client = docker.from_env()

class PlanItem(BaseModel):
    container_id: str
    container_name: str
    image: str
    current_env: Dict[str, str]
    caps: List[str]
    role: str
    desired_name: str
    desired_labels: Dict[str, str]
    port_map: Dict[str, Any]

class Plan(BaseModel):
    colony: str
    items: List[PlanItem]

def emit(topic: str, payload: Dict[str, Any]):
    try:
        with httpx.Client(timeout=6.0) as s:
            s.post(f"{ARCHIVER_URL}/archive", json={
                "source":"organizer","topic":topic,"tags":["organizer","ops"],
                "severity":"INFO","payload":payload
            })
    except Exception:
        pass

def recreate_container(it: PlanItem):
    # Stop and remove existing
    try:
        c = client.containers.get(it.container_id)
        c.stop(timeout=10)
        c.remove()
    except Exception:
        pass
    env = it.current_env.copy()
    env["MICROCELL_NAME"] = it.desired_name
    # carry caps if missing
    if "MICROCELL_CAPS" not in env and it.caps:
        env["MICROCELL_CAPS"] = ",".join(it.caps)
    labels = (it.desired_labels or {})
    # Recreate with same image and ports (host mappings are preserved by docker-py only if specified; for simplicity, run detached and rely on compose/k8s typically)
    container = client.containers.run(it.image, detach=True, environment=env, name=it.desired_name, labels=labels)
    return container.name

@app.get("/health")
def health(): return {"status":"ok","ts": time.time()}

@app.post("/apply", response_model=Dict[str, Any])
def apply(plan: Plan, wire_after: bool = True, start_missing: bool = True):
    results = []
    for it in plan.items:
        try:
            new_name = recreate_container(it)
            results.append({"old": it.container_name, "new": new_name})
            emit("service.renamed", {"old": it.container_name, "new": new_name, "role": it.role})
        except Exception as e:
            results.append({"old": it.container_name, "error": str(e)})
            emit("service.rename_error", {"old": it.container_name, "error": str(e)})
    out = {"renamed": results}
    if wire_after:
        try:
            with httpx.Client(timeout=10.0) as s:
                wr = s.post(f"{WIRE_URL}/wire", json={"require_caps":["router","relay"]}).json()
            out["wiring"] = wr
            emit("wiring.completed", {"caps": wr.get("caps",{}), "missing": wr.get("missing",[]) })
        except Exception as e:
            out["wiring_error"] = str(e)
            emit("wiring.error", {"error": str(e)})
    if start_missing:
        try:
            with httpx.Client(timeout=15.0) as s:
                sr = s.post(f"{STARTER_URL}/start_missing", json={"probe_paths":{"router":"/__cell__/health","relay":"/__cell__/health"}}).json()
            out["starter"] = sr
            emit("starter.completed", {"actions": sr.get("actions",[]), "missing": sr.get("missing",[])})
        except Exception as e:
            out["starter_error"] = str(e)
            emit("starter.error", {"error": str(e)})
    return out
