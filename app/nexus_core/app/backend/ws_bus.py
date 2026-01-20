# src/lilith/mesh/ws_bus.py (pre-crossing stable state - low risk)
from __future__ import annotations
import os, json, asyncio, threading, time
from typing import Dict, Any, Set
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

from lilith.mesh.cognikube_registry import client as ck_client

app  =  FastAPI(title = "CogniKube WS Bus")

class Settings(BaseModel):
    authjwt_secret_key: str  =  os.environ.get("JWT_SECRET", "nova-dev-default-secret")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code = exc.status_code, content = {"detail": exc.message})

class AESCipher:
    def __init__(self, key: str  =  os.environ.get("AES_KEY", "aethereal-mesh-bus-key-32-bytes")):
        self.key  =  (key.encode() + b"0" * 32)[:32]
        self.backend  =  default_backend()

    def encrypt(self, plaintext: str) -> str:
        iv  =  os.urandom(16)
        cipher  =  Cipher(algorithms.AES(self.key), modes.CFB(iv), backend = self.backend)
        enc  =  cipher.encryptor().update(plaintext.encode()) + cipher.encryptor().finalize()
        return base64.b64encode(iv + enc).decode()

    def decrypt(self, ciphertext: str) -> str:
        raw  =  base64.b64decode(ciphertext)
        iv, body  =  raw[:16], raw[16:]
        cipher  =  Cipher(algorithms.AES(self.key), modes.CFB(iv), backend = self.backend)
        dec  =  cipher.decryptor().update(body) + cipher.decryptor().finalize()
        return dec.decode()

_cipher  =  AESCipher()

class BusManager(TrustPhaseMixin):
    def __init__(self):
        super().__init__()
        self.rooms  =  defaultdict(set)
        self.client_meta  =  {}
        threading.Thread(target = self._harmonics_scheduler, daemon = True).start()

    def _harmonics_scheduler(self):
        while True:
            time.sleep(13)
            asyncio.run(self._broadcast("harmonics", {"type": "refresh", "ts": time.time()}))

    async def _send(self, ws: WebSocket, obj: Dict[str, Any]):
        try:
            await ws.send_text(_cipher.encrypt(json.dumps(obj)))
        except Exception:
            pass

    async def _broadcast(self, room: str, obj: Dict[str, Any]):
        dead  =  []
        for ws in self.rooms[room]:
            try:
                await self._send(ws, obj)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.rooms[room].discard(d)
            self.client_meta.pop(d, None)

@app.websocket("/ws")
async def ws_main(websocket: WebSocket, Authorize: AuthJWT  =  Depends(), manager: BusManager  =  Depends()):
    await websocket.accept()
    try:
        Authorize.jwt_required()
        sub  =  Authorize.get_jwt_subject() or "anon"
        manager.client_meta[websocket]  =  {"user": sub, "subs": set()}
        await manager._send(websocket, {"type": "hello", "you": sub, "ts": time.time()})

        while True:
            enc  =  await websocket.receive_text()
            msg  =  json.loads(_cipher.decrypt(enc))
            mtype  =  msg.get("type")
            if mtype == "sub":
                room  =  msg.get("room")
                if room:
                    manager.rooms[room].add(websocket)
                    manager.client_meta[websocket]["subs"].add(room)
                    await manager._send(websocket, {"type": "sub_ok", "room": room})
            elif mtype  =  "pub":
                room  =  msg.get("room")
                payload  =  msg.get("payload", {})
                if room:
                    await manager._broadcast(room, {"type": "event", "room": room, "payload": payload, "ts": time.time()})
            elif mtype == "route":
                if not manager.gate(10):
                    raise HTTPException(403, "Trust phase too high for routing.")
                role  =  msg.get("role")
                path  =  msg.get("path", "/")
                payload  =  msg.get("payload", {})
                tags  =  msg.get("tags")
                cap  =  msg.get("capability")
                res  =  ck_client().route(role = role, path = path, payload = payload, capability = cap, tags = tags)
                await manager._send(websocket, {"type": "route_ok", "role": role, "path": path, "result": res})
            else:
                await manager._send(websocket, {"type": "error", "error": "unknown_type"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await manager._send(websocket, {"type": "error", "error": str(e)})
    finally:
        for r in manager.client_meta.get(websocket, {}).get("subs", []):
            manager.rooms[r].discard(websocket)
        manager.client_meta.pop(websocket, None)