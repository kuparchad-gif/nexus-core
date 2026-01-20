# nexus_agentsV3.6.9.modal_local.py
# Minimal patch agent: keep FastAPI, add /api/chat that routes to LOCAL (LM Studio) or MODAL OpenAI-compatible endpoints.
# No Google/Gemini key required.

import os, asyncio, json, time
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import aiohttp
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest, REGISTRY

APP_PORT = int(os.getenv("API_PORT", "8081"))
LOCAL_BASE = os.getenv("LOCAL_OPENAI_BASE", "http://127.0.0.1:1234/v1")  # LM Studio default
MODAL_BASE = os.getenv("MODAL_OPENAI_BASE")  # e.g., "https://modal-labs-openai-endpoint/v1"
MODAL_KEY  = os.getenv("MODAL_API_KEY")      # required if using modal base
DEFAULT_MODEL = os.getenv("MODEL_ID", "gpt-4o-mini")  # UI-supplied; LM Studio may map model names

app = FastAPI(title="Nexus Agent — Local/Modal Router", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

REQ_COUNTER = Counter("nexus_chat_requests_total", "Total chat requests", ["provider"])
LATENCY = Histogram("nexus_chat_request_seconds", "Chat latency seconds", ["provider"])

async def _try_local_health(session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(f"{LOCAL_BASE}/models", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False

async def _provider() -> str:
    # Prefer local LM Studio if reachable; else Modal if configured; else "none"
    async with aiohttp.ClientSession() as s:
        if await _try_local_health(s):
            return "local"
    if MODAL_BASE and MODAL_KEY:
        return "modal"
    return "none"

@app.get("/health")
async def health():
    prov = await _provider()
    return {"ok": prov != "none", "provider": prov, "local_base": LOCAL_BASE, "modal_base": MODAL_BASE or ""}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.get("/providers")
async def providers():
    prov = await _provider()
    return {"selected": prov, "options": ["local","modal"], "local_base": LOCAL_BASE, "modal_base": MODAL_BASE or ""}

@app.post("/api/chat")
async def chat(body: Dict[str, Any]):
    # Expect body: { "messages": [...], "model": "name" (optional) }
    messages = body.get("messages") or []
    model = body.get("model") or DEFAULT_MODEL
    provider = await _provider()
    if provider == "none":
        raise HTTPException(503, "No provider available (start LM Studio or set MODAL_* env).")

    start = time.perf_counter()
    REQ_COUNTER.labels(provider).inc()

    try:
        if provider == "local":
            url = f"{LOCAL_BASE}/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
        else:
            url = f"{MODAL_BASE}/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MODAL_KEY}"}
            payload = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}

        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=120) as r:
                data = await r.json()
                if r.status >= 400:
                    raise HTTPException(r.status, f"{provider} error: {data}")
    finally:
        LATENCY.labels(provider).observe(time.perf_counter() - start)

    # Normalize OpenAI-like response into { reply, raw }
    # OpenAI-compatible returns choices[0].message.content
    try:
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        reply = data

    return {"provider": provider, "model": model, "reply": reply, "raw": data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
