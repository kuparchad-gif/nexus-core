# Systems/engine/router/src/tensor_router.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, httpx, asyncio, time, json

METATRON_URL = os.getenv("METATRON_URL", "http://metatron:9021/process")
COMPACTIFAI_URL = os.getenv("COMPACTIFAI_URL", "http://compactifai:9031/compress")
GUARDIAN_URL = os.getenv("GUARDIAN_URL", "http://guardian:9040/escalate")  # stub/external
VALIDATOR_URL = os.getenv("VALIDATOR_URL", "http://validator:9033/validate")
METAT_SPINE_URL = os.getenv("METAT_SPINE_URL", "http://metat_spine:9035/wire")

app = FastAPI(title="Tensor Router", version="0.1.0")

class InputMsg(BaseModel):
    id: str
    content: str
    source: str = "client"
    role: str = "user"
    channel: str = "ws"
    meta: dict = {}

async def call_json(url, payload, timeout=3.0):
    async with httpx.AsyncClient(timeout=timeout) as cx:
        r = await cx.post(url, json=payload)
        r.raise_for_status()
        return r.json()

@app.get("/health")
async def health():
    return {"ok": True, "service": "tensor_router"}

@app.post("/ingest")
async def ingest(m: InputMsg):
    # 1) Ask Metatron
    verdict = await call_json(METATRON_URL, {
        "id": m.id, "ts": time.time(), "source": m.source, "role": m.role,
        "channel": m.channel, "content": m.content, "meta": m.meta
    })
    action = verdict.get("action")
    if action == "escalate":
        try:
            await call_json(GUARDIAN_URL, {"id": m.id, "reason": verdict.get("reason"), "meta": m.meta})
        except Exception:
            pass
        raise HTTPException(status_code=451, detail={"status":"escalated","reason":verdict.get("reason")})
    if action in ("drop", "delay"):
        raise HTTPException(status_code=429, detail={"status": action, "reason": verdict.get("reason")})

    # 1.5) Wire signal through MetatSpine
    trust_score = verdict.get("trust_score", 0.6)
    # Convert content to a signal of floats
    signal = [float(b) / 255.0 for b in m.content.encode('utf-8')]
    # Phase is derived from trust score, from 0 to 30
    phase = int(trust_score * 30)

    spine_payload = {"signal": signal, "phase": phase}
    spine_response = await call_json(METAT_SPINE_URL, spine_payload)

    # Convert the processed signal back to a string
    processed_signal = spine_response.get("output", [])
    processed_content_bytes = bytes([int(f * 255) for f in processed_signal if 0 <= f <= 1])
    processed_content = processed_content_bytes.decode('utf-8', errors='ignore')


    # 2) Permit path → CompactifAI (compute compression profile if needed)
    payload = {
        "id": m.id, "content": processed_content,
        "trust_score": trust_score,
        "priority": 1.0 - verdict.get("shaped_priority", 0.5),
        "load": verdict.get("passthrough", {}).get("load", 0.0),
        "compression_profile": verdict.get("passthrough", {}).get("compression_profile")
    }
    cp = await call_json(COMPACTIFAI_URL, payload)

    # 3) Route to LoRAMoE (we just forward compacted request)
    async with httpx.AsyncClient(timeout=3.0) as cx:
        r = await cx.post("http://lora_moe:9032/infer", json={"id": m.id, "content": processed_content, "compression_profile": cp.get("compression_profile")})
        r.raise_for_status()
        model_out = r.json()

    # 4) Validate → may trigger AutoHealing
    result = await call_json(VALIDATOR_URL, {"id": m.id, "input": processed_content, "output": model_out.get("output","")})
    return {"ok": True, "verdict": verdict, "compression": cp, "model_out": model_out, "validator": result}
