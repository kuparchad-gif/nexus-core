# src/lilith/spine/relay_pi.py
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json, asyncio, hmac, hashlib, os

from src.lilith.metatron.filter_pi import MetatronFilterPI
from src.lilith.policy.trust_phase import TrustPhaseMixin

# Mesh hooks (assumed to exist per scaffold)
try:
    from src.lilith.mesh.bus import WsBusClient
    from src.lilith.mesh.mcp_adapter import MCPAdapter
    from src.lilith.mesh.registry import RegistryClient
except Exception:
    WsBusClient = MCPAdapter = RegistryClient = None  # graceful fallback for initial bring-up

JWT_SECRET = os.getenv("NEXUS_JWT_SECRET", "dev-secret")
WS_AES_KEY = os.getenv("NEXUS_AES_KEY", "dev-aes-key")  # provided to bus client if needed

app = FastAPI(title="Lillith Spine (π-relay)")

class RelayIn(BaseModel):
    signal: List[float]  # len=13
    step: int = 0

class LillithSpine(TrustPhaseMixin):
    def __init__(self):
        super().__init__()
        self.filter = MetatronFilterPI()
        self.bus = None
        self.mcp = None
        self.registry = None

    def init_mesh(self):
        if RegistryClient:
            self.registry = RegistryClient(service="lillith-spine", capabilities=["relay","policy"], tags=["pi","metatron"])
            self.registry.register()
        if WsBusClient:
            self.bus = WsBusClient(service="lillith-spine", jwt_secret=JWT_SECRET, aes_key=WS_AES_KEY)
        if MCPAdapter:
            self.mcp = MCPAdapter(service="lillith-spine", bus=self.bus)

L = LillithSpine()
L.init_mesh()

def _verify_jwt(authorization: Optional[str]) -> None:
    # Minimal HMAC "JWT-like" check (replace with real JWT lib in your stack)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ",1)[1].strip()
    sig = hmac.new(JWT_SECRET.encode(), b"AUTH", hashlib.sha256).hexdigest()
    if not hmac.compare_digest(token, sig):
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/policy")
def policy(authorization: Optional[str] = Header(default=None)):
    _verify_jwt(authorization)
    return L.policy_dict()

@app.post("/relay")
def relay(payload: RelayIn, authorization: Optional[str] = Header(default=None)):
    _verify_jwt(authorization)
    y = L.filter.apply(payload.signal, payload.step)
    # Publish on bus (MCP envelope over WS) if available
    if L.bus and L.mcp:
        env = {"type": "mcp.signal", "from": "lillith-spine", "signal": y, "step": payload.step}
        L.mcp.send(env)
    return {"signal": y, "step": payload.step}

# WebSocket entry (JWT/AES handled by bus client—here we only gate initial auth)
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        first = await ws.receive_text()
        data = json.loads(first)
        token = data.get("token","")
        sig = hmac.new(JWT_SECRET.encode(), b"AUTH", hashlib.sha256).hexdigest()
        if not hmac.compare_digest(token, sig):
            await ws.close(code=4401)
            return
        # Hand off to bus if available; else simple echo (policy-gated tools opening)
        if L.bus:
            await L.bus.attach(ws)
        else:
            await ws.send_text(json.dumps({"ok": True, "policy": L.policy_dict()}))
            while True:
                msg = await ws.receive_text()
                await ws.send_text(msg)
    except WebSocketDisconnect:
        return
