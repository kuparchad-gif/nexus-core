from __future__ import annotations
import os, json, asyncio, uuid, time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from pydantic import BaseModel
from collections import defaultdict

# Simple in-proc event bus
class EventBus:
    def __init__(self):
        self._subs = defaultdict(list)
    def subscribe(self, topic: str, fn: Callable[[Dict[str,Any]], None]):
        self._subs[topic].append(fn)
    async def publish(self, topic: str, msg: Dict[str,Any]):
        for fn in self._subs.get(topic, []):
            try:
                r = fn(msg)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

bus = EventBus()

@dataclass
class EdgeMessage:
    id: str
    ts: float
    src: str
    topic: str
    payload: Dict[str, Any]
    def dict(self): return asdict(self)

class RouteRequest(BaseModel):
    topic: str
    payload: Dict[str, Any] = {}
    src: str = "unknown"

app = FastAPI(title="Edge AnyNode")

# ws clients by topic
ws_clients: Dict[str, list] = defaultdict(list)

@app.on_event("startup")
async def _boot():
    # Example internal handler: echo diagnostics
    def diag_handler(msg: Dict[str,Any]):
        # Could push into Loki or Heart
        return None
    bus.subscribe("diagnostic", diag_handler)

@app.post("/edge/route")
async def route(req: RouteRequest):
    msg = EdgeMessage(
        id=str(uuid.uuid4()), ts=time.time(), src=req.src, topic=req.topic, payload=req.payload
    )
    await bus.publish(req.topic, msg.dict())
    # also relay to WebSocket listeners
    for ws in list(ws_clients.get(req.topic, [])):
        try:
            await ws.send_json({"type":"edge_event","data":msg.dict()})
        except Exception:
            try:
                ws_clients[req.topic].remove(ws)
            except Exception:
                pass
    return {"ok": True, "edge_id": msg.id}

@app.websocket("/edge/ws/{topic}")
async def edge_ws(websocket: WebSocket, topic: str):
    await websocket.accept()
    ws_clients[topic].append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Allow clients to publish via ws too
            if isinstance(data, dict) and data.get("type") == "publish":
                await bus.publish(topic, data.get("payload", {}))
    except WebSocketDisconnect:
        ws_clients[topic].remove(websocket)

# Lightweight registrar so services can announce capabilities
_registry: Dict[str, Dict[str, Any]] = {}

class RegisterRequest(BaseModel):
    service: str
    version: str = "0.1"
    topics: list[str] = []

@app.post("/edge/register")
async def register(req: RegisterRequest):
    _registry[req.service] = {"version": req.version, "topics": req.topics, "ts": time.time()}
    return {"ok": True, "registry": _registry}

@app.get("/edge/registry")
async def registry():
    return {"ok": True, "registry": _registry}
