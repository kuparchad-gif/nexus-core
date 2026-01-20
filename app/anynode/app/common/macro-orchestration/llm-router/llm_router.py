# Minimal LLM router: forwards JSON to BACKEND_URL (Ollama-compatible /api/generate fallback)
import os, json, socket
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx

MODEL = os.getenv("LLM_MODEL","qwen2.5:1.5b")
BACKEND_URL = os.getenv("BACKEND_URL","[REDACTED-URL])  # Ollama default
PORT = int(os.getenv("PORT","8009"))

app = FastAPI(title="LLM Router", version="0.1")

@app.get("/healthz")
def health():
    return {"ok": True, "model": MODEL, "backend": BACKEND_URL, "host": socket.gethostname()}

@app.post("/v1/chat")
async def chat(payload: dict):
    # If BACKEND_URL looks like Ollama, translate to /api/generate
    if "11434" in BACKEND_URL or BACKEND_URL.endswith("/api/generate"):
        req = {
            "model": MODEL,
            "prompt": payload.get("prompt") or payload.get("messages", [{}])[-1].get("content",""),
            "stream": False
        }
        target = BACKEND_URL if BACKEND_URL.endswith("/api/generate") else (BACKEND_URL + "/api/generate")
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(target, json=req)
            r.raise_for_status()
            data = r.json()
            text = data.get("response") or data.get("message",{}).get("content","")
            return {"model": MODEL, "output": text}
    else:
        # Generic passthrough
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(BACKEND_URL, json=payload)
            r.raise_for_status()
            return JSONResponse(r.json())

@app.post("/v1/generate")
async def generate(payload: dict):
    return await chat(payload)

