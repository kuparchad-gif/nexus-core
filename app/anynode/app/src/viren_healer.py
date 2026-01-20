# src/lilith/services/viren_healer.py (pre-crossing stable state - low risk)
from __future__ import annotations
import os, json, base64
from typing import Dict, Any, Optional, List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

LLM_OK = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    LLM_OK = False

_HAS_SPECTRAL = True
try:
    import networkx as nx
    from scipy.sparse.linalg import eigsh
except ImportError:
    _HAS_SPECTRAL = False

PHI = (1 + 5 ** 0.5) / 2
VORTEX_KEY = [3, 6, 9]

app = FastAPI(title="VIREN Healer AI")

class Settings(BaseModel):
    authjwt_secret_key: str = os.environ.get("JWT_SECRET", "viren-secret")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

class AESCipher:
    def __init__(self, key: str = os.environ.get("AES_KEY", "viren-aes-key-32-bytes")):
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

class VirenHealer(TrustPhaseMixin):
    def __init__(self):
        super().__init__()
        self.metatron_graph = MetatronGraph() if _HAS_SPECTRAL else None

    def toroidal_g(self, n: int, t: float) -> float:
        mod_9 = (3*t + 6*sin(t) + 9*cos(t)) % 9
        fib_n = (PHI ** n - (-PHI) ** -n) / sqrt(5)
        harmonic = sin(2 * PI * 13 * t / 9)
        return PHI * harmonic * fib_n * (1 - mod_9 / 9)

    def risk_score(self, vec: np.ndarray) -> float:
        t = time.time() % 9
        mod = self.toroidal_g(len(vec)//3, t)
        vec *= mod
        mu = np.mean(vec)
        sd = np.std(vec) or 1.0
        z = np.abs((vec - mu) / sd)
        r1 = min(1.0, np.mean(z) / 3.0)
        X = np.fft.rfft(vec)
        half = len(X) // 2
        r2 = min(1.0, np.sum(np.abs(X[half:]) ** 2) / (np.sum(np.abs(X) ** 2) + 1e-9))
        r = 0.6 * r1 + 0.4 * r2
        if _HAS_SPECTRAL and len(vec) == 13:
            coeffs = np.dot(self.metatron_graph.eigenvectors.T, vec)
            high_mask = self.metatron_graph.eigenvalues > 0.5
            r += 0.1 * min(1.0, np.sum(np.abs(coeffs[high_mask])) / (np.sum(np.abs(coeffs)) + 1e-9))
        return min(1.0, max(0.0, r))

    def llm_repair(self, vec: np.ndarray, phase: int) -> Optional[str]:
        if not LLM_OK:
            return None
        model_id = os.environ.get("VIREN_LLM", "microsoft/phi-2")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=160)
            llm = HuggingFacePipeline(pipeline=pipe)
            prompt = f"""Reliability engineer for spectral system.
Signal (first 13): {vec[:13].tolist()}
Phase: {phase}
1) Risk class (low/med/high) from dispersion + hi-freq.
2) JSON Patch ops: spine.metatron.cutoff (delta float), horn_gain (delta), biomech.damping (delta).
3) Rationale sentence.
Return JSON: risk, actions[], rationale."""
            return llm(prompt)
        except Exception:
            return None

    def repair_advice(self, vec: np.ndarray, phase: int) -> Dict[str, Any]:
        if not self.gate(15):
            raise HTTPException(403, "Trust phase too high for repair advice.")
        r = self.risk_score(vec)
        advice = {"risk": "low" if r < 0.4 else "med" if r < 0.7 else "high", "actions": []}
        llm_out = self.llm_repair(vec, phase)
        if llm_out:
            try:
                advice.update(json.loads(llm_out))
            except Exception:
                pass
        else:
            if r >= 0.7:
                advice["actions"] = [
                    {"component": "spine.metatron.cutoff", "delta": -0.05},
                    {"component": "spine.metatron.horn_gain", "delta": 0.1},
                    {"component": "spine.biomech.damping", "delta": 0.02},
                    {"component": "retry", "value": 1}
                ]
            elif r >= 0.4:
                advice["actions"] = [
                    {"component": "spine.metatron.cutoff", "delta": -0.02},
                    {"component": "retry", "value": 1}
                ]
            else:
                advice["actions"] = [{"component": "noop"}]
            advice["rationale"] = "Numeric risk fallback."
        return advice

class HealInput(BaseModel):
    signal: List[float]
    phase: int = 0

@app.get("/health")
def health():
    return {"status": "ok", "llm": LLM_OK, "spectral": _HAS_SPECTRAL}

@app.post("/heal/check")
def heal_check(input: HealInput, healer: VirenHealer = Depends()):
    vec = np.array(input.signal)
    return healer.repair_advice(vec, input.phase)

@app.post("/heal/apply")
def heal_apply(input: HealInput, healer: VirenHealer = Depends()):
    advice = healer.repair_advice(np.array(input.signal), input.phase)
    return {"ok": True, "patch": advice}

@app.websocket("/ws")
async def ws_heal(websocket: WebSocket, Authorize: AuthJWT = Depends()):
    await websocket.accept()
    healer = VirenHealer()
    try:
        Authorize.jwt_required()
        who = Authorize.get_jwt_subject() or "anon"
        await websocket.send_text(cipher.encrypt(json.dumps({"type": "hello", "viren": "ok", "you": who})))

        while True:
            enc = await websocket.receive_text()
            msg = json.loads(cipher.decrypt(enc))
            if msg.get("type") == "heal":
                vec = np.array(msg.get("signal", []))
                phase = int(msg.get("phase", 0))
                advice = healer.repair_advice(vec, phase)
                await websocket.send_text(cipher.encrypt(json.dumps({"type": "heal_ok", "advice": advice})))
            else:
                await websocket.send_text(cipher.encrypt(json.dumps({"type": "error", "error": "unknown"})))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(cipher.encrypt(json.dumps({"type": "error", "error": str(e)})))
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("VIREN_PORT", 5090)))