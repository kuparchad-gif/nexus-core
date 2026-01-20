from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, time, docker, re, httpx, json

app = FastAPI(title="SurveyorBot", version="0.1.0")

# External services (optional)
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8500")
WIRE_URL     = os.getenv("WIRE_URL", "http://localhost:9021")  # WireCrawler
STARTER_URL  = os.getenv("STARTER_URL", "http://localhost:9022") # StarterCrawler
ARCHIVER_URL = os.getenv("ARCHIVER_URL", "http://localhost:9020")
LOKI_URL     = os.getenv("LOKI_URL", "http://localhost:9011")

client = docker.from_env()

ROLE_BY_CAP = {
    "archive": "archiver",
    "search": "archiver",
    "timeline": "archiver",
    "router": "consciousness",
    "chat": "consciousness",
    "embed": "consciousness",
    "relay": "nexus-core",
    "pi": "nexus-core",
    "healer": "viren",
    "autopatch": "viren",
    "risk": "viren",
    "forensics": "loki",
    "logs": "loki",
    "crawler": "crawler",
    "wire": "crawler-wire",
    "starter": "crawler-starter",
    "anynode": "anynode"
}

class PlanItem(BaseModel):
    container_id: str
    container_name: str
    image: str
    current_env: Dict[str, str]
    caps: List[str] = Field(default_factory=list)
    role: str
    desired_name: str
    desired_labels: Dict[str, str]
    port_map: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    colony: str = "alpha"
    items: List[PlanItem] = Field(default_factory=list)

def parse_env(env_list: List[str]) -> Dict[str, str]:
    out = {}
    for e in env_list or []:
        if "=" in e:
            k, v = e.split("=", 1)
            out[k] = v
    return out

def classify_role(caps: List[str]) -> str:
    for c in caps:
        if c in ROLE_BY_CAP:
            return ROLE_BY_CAP[c]
    # fallback
    return "anynode"

def desired_name(role: str, colony: str, index: int) -> str:
    return f"{role}-{colony}-{index:02d}"

def collect_ports(container) -> Dict[str, Any]:
    ports = container.attrs.get("NetworkSettings", {}).get("Ports", {}) or {}
    out = {}
    for k, v in ports.items():
        out[k] = v
    return out

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/survey", response_model=Plan)
def survey(colony: str = "alpha"):
    containers = client.containers.list(all=True)
    buckets: Dict[str, List[PlanItem]] = {}
    for c in containers:
        env = parse_env(c.attrs.get("Config", {}).get("Env", []))
        caps = [x.strip() for x in (env.get("MICROCELL_CAPS","") or "").split(",") if x.strip()]
        role = classify_role(caps) if caps else "anynode"
        idx = len(buckets.get(role, [])) + 1
        name = desired_name(role, colony, idx)
        labels = {"role": role, "colony": colony}
        item = PlanItem(
            container_id=c.id[:12],
            container_name=c.name,
            image=c.attrs.get("Config", {}).get("Image",""),
            current_env=env,
            caps=caps,
            role=role,
            desired_name=name,
            desired_labels=labels,
            port_map=collect_ports(c)
        )
        buckets.setdefault(role, []).append(item)
    items = []
    for role, arr in buckets.items():
        for i, it in enumerate(arr, start=1):
            it.desired_name = desired_name(role, colony, i)
            items.append(it)
    plan = Plan(colony=colony, items=items)
    return plan

@app.post("/plan")
def plan(colony: str = "alpha"):
    return survey(colony)

@app.post("/wire_preview")
def wire_preview(require_caps: List[str] = ["router","relay"]):
    try:
        with httpx.Client(timeout=10.0) as s:
            r = s.post(f"{WIRE_URL}/wire", json={"require_caps": require_caps})
            return r.json()
    except Exception as e:
        return {"error": str(e)}
