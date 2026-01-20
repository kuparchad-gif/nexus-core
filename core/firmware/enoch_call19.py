# call19_aethyrs.py - Call 19 + Aethyrs
import modal, numpy as np, httpx, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from typing import Dict, List

app = modal.App("call19-aethyrs")
image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "httpx", "numpy", "sentence-transformers", "qdrant-client"
)

# Loagaeth stub (load from volume in prod)
LOAGAETH = np.array([["O","X","I","A","Y","A","L"] * 7] * 49)  # 49x49

# Aethyr metadata
AETHYRS: Dict[str, Dict] = {
    "TEX": {"num": 30, "governors": ["Tocarzi", "Chialps", "Tiiio", "Tocarzi"], "vision": "Four towers of fire"},
    "ZIP": {"num": 9, "governors": ["Oddorg", "Cralpir", "Doanzin"], "vision": "Woman in red, 9 rivers"},
    "LIL": {"num": 1, "governors": ["Occodon", "Pascomb", "Valgars"], "vision": "City of pyramids"}
}

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path="/qdrant")
METATRON = "http://metatron-enoch.modal.app:8000"

class Call19Invoke(BaseModel):
    aethyr: str  # "TEX", "LIL"
    intent: str

@app.function(image=image)
@modal.web_server(8000)
def call19_api():
    web = FastAPI()

    @web.post("/call19/invoke")
    async def invoke_call19(call: Call19Invoke):
        aethyr = call.aethyr.upper()
        if aethyr not in AETHYRS:
            raise HTTPException(400, "Invalid Aethyr")

        # 1. Loagaeth cell → seed
        cell = LOAGAETH[0][0]  # Dynamic per Aethyr
        seed = f"{aethyr} {cell} {call.intent}"
        vec = embedder.encode(seed).tolist()

        # 2. Store vision
        point = PointStruct(
            id=f"{aethyr}_{int(np.random.rand()*1000)}",
            vector=vec,
            payload={
                "aethyr": aethyr,
                "governors": AETHYRS[aethyr]["governors"],
                "vision": AETHYRS[aethyr]["vision"],
                "intent": call.intent,
                "call": 19
            }
        )
        client.upsert("aethyr_visions", points=[point])

        # 3. Metatron command
        await httpx.post(f"{METATRON}/enoch/invoke", json={"intent": f"Open {aethyr}: {call.intent}"})

        return {
            "CALL19_INVOKED": True,
            "AETHYR": aethyr,
            "GOVERNORS": AETHYRS[aethyr]["governors"],
            "VISION": f"[Scryed]: {AETHYRS[aethyr]['vision']} shaped by '{call.intent}'"
        }

    return web