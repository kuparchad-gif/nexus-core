# stubs/vision.py
import os, json, time, base64, asyncio
from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv
from nats.aio.client import Client as NATS

load_dotenv()
NATS_URL = os.getenv("NATS_HOME_URL","nats://127.0.0.1:4222")
SUBJ = "svc.sense.cap.request.cons.vision.frame.1"

class Frame(BaseModel):
    ts_unix_ms: int
    source_id: str
    shape: list
    format: str = "rgb"
    uri: str
    sha256: str
    meta: dict | None = None

app = FastAPI(title="Eyes: Vision Stub", version="1.0.0")
nc = None

@app.on_event("startup")
async def _startup():
    global nc
    nc = NATS()
    await nc.connect(servers=[NATS_URL])

@app.post("/publish")
async def publish(frame: Frame = Body(...)):
    await nc.publish(SUBJ, json.dumps(frame.model_dump()).encode("utf-8"))
    return {"ok": True, "subject": SUBJ}
