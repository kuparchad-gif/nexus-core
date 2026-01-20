# src/lilith/services/loki_ai.py (pre-crossing stable state - low risk)
from __future__ import annotations
import os, time, json
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

LLM_OK = True
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    LLM_OK = False

app = FastAPI(title="Loki Sentinel AI")

class Settings(BaseModel):
    authjwt_secret_key: str = os.environ.get("JWT_SECRET", "loki-secret")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

class AESCipher:
    def __init__(self, key: str = os.environ.get("AES_KEY", "loki-aes-key-32-bytes")):
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

cipher = AESCipher()

class LokiSentinel:
    def __init__(self):
        self.ring = deque(maxlen=int(os.environ.get("LOKI_RING_CAP", 10000)))
        self.stats = defaultdict(int)

    def log_event(self, ev: Dict[str, Any]) -> None:
        ev = dict(ev)
        ev["ts"] = time.time()
        self.ring.append(ev)
        self.stats[ev.get("level", "info")] += 1

    def llm_explain(self, event: Dict[str, Any]) -> Optional[str]:
        if not LLM_OK:
            return None
        model_id = os.environ.get("LOKI_LLM", "microsoft/phi-2")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=120)
            llm = HuggingFacePipeline(pipeline=pipe)
            prompt = f"Explain impact and 2 triage steps for event:\n{json.dumps(event)}"
            return llm(prompt)
        except Exception:
            return None

    def analyze(self, window_s: float = 60.0, threshold: int = 50) -> Dict[str, Any]:
        now = time.time()
        recent = [r for r in self.ring if (now - r["ts"]) <= window_s]
        counts = defaultdict(int)
        for r in recent:
            k = f'{r.get("kube", "unknown")}:{r.get("level", "info")}'
            counts[k] += 1
        alerts = [{"kube_level": k, "count": c} for k, c in counts.items() if c >= threshold]
        return {"window": window_s, "threshold": threshold, "alerts": alerts}

loki = LokiSentinel()

class LogInput(BaseModel):
    event: Dict[str, Any]

@app.post("/forensics/log")
def log_event(input: LogInput):
    loki.log_event(input.event)
    return {"ok": True, "size": len(loki.ring)}

class AnalyzeInput(BaseModel):
    window_s: float = 60.0
    threshold: int = 50

@app.post("/forensics/analyze")
def analyze(input: AnalyzeInput):
    return loki.analyze(input.window_s, input.threshold)

@app.websocket("/ws")
async def ws(websocket: WebSocket, Authorize: AuthJWT = Depends()):
    await websocket.accept()
    try:
        Authorize.jwt_required()
        who = Authorize.get_jwt_subject() or "anon"
        await websocket.send_text(cipher.encrypt(json.dumps({"type": "hello", "loki": "ok", "you": who})))
        while True:
            enc = await websocket.receive_text()
            msg = json.loads(cipher.decrypt(enc))
            if msg.get("type") == "explain":
                out = loki.llm_explain(msg.get("event", {})) or "LLM disabled."
                await websocket.send_text(cipher.encrypt(json.dumps({"type": "explain_ok", "text": out})))
            elif msg.get("type") == "log":
                loki.log_event(msg.get("event", {}))
                await websocket.send_text(cipher.encrypt(json.dumps({"type": "log_ok"})))
            else:
                await websocket.send_text(cipher.encrypt(json.dumps({"type": "error", "error": "unknown"})))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(cipher.encrypt(json.dumps({"type": "error", "error": str(e)})))
        await websocket.close()