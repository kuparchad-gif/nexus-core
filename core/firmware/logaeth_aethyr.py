# loagaeth_aethyr.py - Tables + Aethyrs
import modal,np,httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app=modal.App("loagaeth-aethyr")
image=modal.Image.debian_slim().pip_install("fastapi uvicorn httpx numpy sentence-transformers")
embedder=SentenceTransformer("all-MiniLM-L6-v2")
client=QdrantClient(path="/qdrant")
METATRON="http://metatron-enoch.modal.app:8000"

# Loagaeth stub (49x49)
LOAGAETH_TABLE1=np.array([["I","A","M",...]*49]*49) # Full in volume

class AethyrCall(BaseModel):
    aethyr:str # "TEX"
    intent:str

@app.function(image=image)
@modal.web_server(8000)
def aethyr_api():
    web=FastAPI()
    @web.post("/aethyr/{name}")
    async def explore(name:str.upper(),call:AethyrCall):
        if name not in ["TEX","LIL"]: return {"error":"Invalid Aethyr"}
        # Loagaeth cell → embed
        cell=LOAGAETH_TABLE1[0][0] # Dynamic later
        vec=embedder.encode(f"{call.intent} {cell}").tolist()
        client.upsert("loagaeth_core",[{"id":name,"vector":vec,"payload":{"aethyr":name,"intent":call.intent}}])
        await httpx.post(f"{METATRON}/enoch/invoke",json={"intent":f"Open {name}: {call.intent}"})
        return{"AETHYR_OPENED":name,"vision":"[Angelic JSON]"}
    return web