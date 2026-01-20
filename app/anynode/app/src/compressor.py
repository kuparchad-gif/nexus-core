# Systems/services/compactifai/src/compressor.py
from fastapi import FastAPI
from pydantic import BaseModel
import math

app = FastAPI(title="CompactifAI Compressor", version="0.1.0")

class Req(BaseModel):
    id: str
    content: str
    trust_score: float = 0.6
    priority: float = 0.5
    load: float = 0.0
    compression_profile: dict | None = None

def adaptive_chi(trust: float, load: float, depth: int = 12):
    # Produce per-layer χ, higher at early layers for low trust; lower (more compression) for high trust/load
    prof = {}
    for layer in range(depth):
        base = 64 - int(56 * trust)   # 64..8
        load_push = 8 + int(16 * load)  # compress deeper under load
        if layer < 4 and trust < 0.6:
            chi = max(32, base)       # protect early layers on low trust
        else:
            chi = max(8, min(64, base + (layer//3) - load_push//4))
        prof[f"L{layer}"] = int(chi)
    return prof

@app.get("/health")
def health():
    return {"ok": True, "service": "compactifai"}

@app.post("/compress")
def compress(req: Req):
    cp = req.compression_profile or adaptive_chi(req.trust_score, req.load)
    return {"compression_profile": cp}
