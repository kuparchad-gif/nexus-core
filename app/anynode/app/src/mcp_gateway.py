# Minimal MCP gateway: subscribes to NATS subject prefix and logs msgs
import os, asyncio, sys
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import socket

try:
    import nats
except Exception as e:
    nats = None

NATS_URL = os.getenv("NATS_URL","")
SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX","mcp.default.")
PORT = int(os.getenv("PORT","8010"))
HOSTNAME = socket.gethostname()

app = FastAPI(title="MCP Gateway", version="0.1")

@app.get("/healthz", response_class=PlainTextResponse)
def health():
    ok = bool(nats) and bool(NATS_URL)
    return "ok" if ok else "degraded"

async def nats_worker():
    if not nats or not NATS_URL:
        print("[mcp-gateway] nats not configured; idle")
        return
    nc = await nats.connect(servers=[NATS_URL], tls=(os.getenv("NATS_TLS","0")=="1" or NATS_URL.startswith("tls://")))
    subj = f"{SUBJECT_PREFIX}>"
    async def handler(msg):
        print(f"[{HOSTNAME}] {msg.subject}: {msg.data.decode()[:200]}")
    await nc.subscribe(subj, cb=handler)
    print(f"[mcp-gateway] subscribed to {subj} on {NATS_URL}")
    while True:
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    asyncio.create_task(nats_worker())


