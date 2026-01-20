# src/lilith/spine/ws_spine.py (pre-crossing stable state - low risk)
from __future__ import annotations
import os, json, base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from nexus_spinal_elemental import NexusSpinal

app = FastAPI(title="Spine+WS")

class Settings(BaseModel):
    authjwt_secret_key: str = os.environ.get("JWT_SECRET", "spine-ws-secret")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

class AESCipher:
    def __init__(self, key: str = os.environ.get("AES_KEY", "spine-ws-bridge-key-32")):
        self.key = (key.encode() + b"0" * 32)[:32]
        self.backend = default_backend()

    def encrypt(self, plaintext: str) -> str:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=self.backend)
        enc = cipher.encryptor().update(plaintext.encode()) + cipher.encryptor().finalize()
        return base64.b64encode(iv + enc).decode()

    def decrypt(self, ciphertext: str) -> str:
        raw = base64.b64decode(ciphertext)
        iv, body = raw[:16], raw[16:]
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=self.backend)
        dec = cipher.decryptor().update(body) + cipher.decryptor().finalize()
        return dec.decode()

_cipher = AESCipher()

@app.websocket("/ws")
async def ws_spine(websocket: WebSocket, Authorize: AuthJWT = Depends()):
    await websocket.accept()
    spinal = NexusSpinal()
    try:
        Authorize.jwt_required()
        who = Authorize.get_jwt_subject() or "anon"
        await websocket.send_text(_cipher.encrypt(json.dumps({"type": "hello", "spine": "ok", "you": who})))

        while True:
            enc = await websocket.receive_text()
            msg = json.loads(_cipher.decrypt(enc))
            if msg.get("type") == "relay":
                signal = msg.get("signal", [])
                phase = msg.get("phase", 1)
                output = spinal.relay_wire(signal, phase)
                await websocket.send_text(_cipher.encrypt(json.dumps({"type": "relay_ok", "output": output})))
            else:
                await websocket.send_text(_cipher.encrypt(json.dumps({"type": "error", "error": "unknown"})))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(_cipher.encrypt(json.dumps({"type": "error", "error": str(e)})))
        await websocket.close()