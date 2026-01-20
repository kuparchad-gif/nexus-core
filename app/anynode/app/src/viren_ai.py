# src/lilith/services/viren_ai.py
from __future__ import annotations
import os, json, base64, time
from typing import Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# optional LLM
LLM_OK = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
except Exception:
    LLM_OK = False

app = FastAPI(title="VIREN Healer AI")

class Settings(BaseModel):
    authjwt_secret_key: str = os.environ.get("JWT_SECRET","viren-secret")

@AuthJWT.load_config
def get_config(): return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

class AESCipher:
    def __init__(self, key: bytes):
        if isinstance(key, str):
            key = key.encode()
        self.key = (key + b"0"*32)[:32]
        self.backend = default_backend()

    def encrypt(self, plaintext: str) -> str:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=self.backend)
        enc = cipher.encryptor().update(plaintext.encode()) + cipher.encryptor().finalize()
        return base64.b64encode(iv + enc).decode()

    def decrypt(self, ciphertext: str) -> str:
        raw = base64.b64decode(ciphertext.encode())
        iv, body = raw[:16], raw[16:]
        cipher = Cipher(algorithms.AES(self.key), modes)
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=self.backend)
        dec = cipher.decryptor().update(body) + cipher.decryptor().finalize()
        return dec.decode()

cipher = AESCipher(os.environ.get("AES_KEY","viren-aes-key-32-bytes"))

def risk_score(vec: np.ndarray) -> float:
    mu = float(np.mean(vec)); sd = float(np.std(vec)) or 1.0
    z = np.abs((vec - mu)/sd)
    r1 = min(1.0, float(np.mean(z)/3.0))
    X = np.fft.rfft(vec); half = len(X)//2; hi = X[half:]
    total = np.sum(np.abs(X)**2) + 1e-9
    r2 = min(1.0, float(np.sum(np.abs(hi)**2)/total))
    return float(0.6*r1 + 0.4*r2)

def llm_repair(vec: np.ndarray, phase: int) -> Optional[str]:
    if not LLM_OK: return None
    try:
        model_id = os.environ.get("VIREN_LLM","microsoft/phi-2")
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=160)
        llm = HuggingFacePipeline(pipeline=pipe)
        prompt = f"""You are a reliability engineer for a spectral control system.
Signal vector (first 13 shown): {vec[:13].tolist()}
Phase: {phase}
1) Estimate risk class (low/med/high) from dispersion + high-frequency energy.
2) Propose patch operations in JSON Patch-like ops targeting:
   - spine.metatron.cutoff (float delta)
   - spine.metatron.horn_gain (float delta)
   - spine.biomech.damping (float delta)
3) Include a one-sentence rationale.
Return only JSON with fields: risk, actions[], rationale."""
        return llm(prompt)
    except Exception:
        return None

@app.get("/health")
def health():
    return {"status":"ok","llm": LLM_OK}

@app.websocket("/ws")
async def ws(websocket: WebSocket, Authorize: AuthJWT = Depends()):
    await websocket.accept()
    try:
        Authorize.jwt_required()
        who = Authorize.get_jwt_subject() or "anon"
        await websocket.send_text(cipher.encrypt(json.dumps({"type":"hello","viren":"ok","you":who})))

        while True:
            enc = await websocket.receive_text()
            msg = json.loads(cipher.decrypt(enc))
            if msg.get("type") == "heal":
                vec = np.array(msg.get("signal",[]), dtype=float)
                phase = int(msg.get("phase",0))
                r = risk_score(vec)
                tool = llm_repair(vec, phase) or json.dumps({
                    "risk": "low" if r<0.4 else "med" if r<0.7 else "high",
                    "actions": [{"component":"retry","value":1}],
                    "rationale": "LLM disabled; using numeric risk."
                })
                await websocket.send_text(cipher.encrypt(json.dumps({"type":"heal_ok","risk":r,"llm":tool})))
            else:
                await websocket.send_text(cipher.encrypt(json.dumps({"type":"error","error":"unknown"})))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(cipher.encrypt(json.dumps({"type":"error","error":str(e)})))
        await websocket.close()
