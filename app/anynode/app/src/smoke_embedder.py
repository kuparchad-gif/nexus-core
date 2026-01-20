from fastapi import FastAPI
from pydantic import BaseModel
import hashlib

app = FastAPI(title="SmokeEmbedder")

class Req(BaseModel):
    text: str
    dim: int = 384

@app.post("/embed")
def embed(r: Req):
    h = hashlib.sha256(r.text.encode()).digest()
    # repeat hash bytes until we hit dim, map 0..255 → -1..+1
    b = (h * ((r.dim // len(h)) + 1))[:r.dim]
    vec = [(x - 128) / 128.0 for x in b]
    return {"embedding": vec, "dim": r.dim, "model": "smoke-embedder"}
