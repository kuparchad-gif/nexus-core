from fastapi import FastAPI

import os, ssl, asyncio
import nats, httpx

app  =  FastAPI()
_nc  =  None

def _tls_ctx():
    ctx  =  ssl.create_default_context(cafile = "/certs/ca/ca.crt")
    ctx.load_cert_chain("/certs/client/client.crt","/certs/client/client.key")
    ctx.minimum_version  =  ssl.TLSVersion.TLSv1_2
    return ctx

@app.on_event("startup")
async def _bus_connect():
    global _nc
    url  =  os.getenv("MCP_NATS_URL", "nats://host.containers.internal:4222")
    _nc  =  await nats.connect(servers = [url], tls = _tls_ctx())

@app.on_event("shutdown")
async def _bus_close():
    if _nc:
        await _nc.drain()

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/api/ping")
def ping():
    return {"ping": "pong"}

@app.get("/api/llm/info")
async def llm_info():
    base  =  os.getenv("LLM_BASE_URL", "http://ollama:11434")
    try:
        async with httpx.AsyncClient(timeout = 5) as cx:
            r  =  await cx.get(f"{base}/api/tags")
        return {"llm_base": base, "status": r.status_code}
    except Exception as e:
        return {"llm_base": base, "error": str(e)}

@app.get("/api/mcp/ping")
async def mcp_ping():
    try:
        msg  =  await _nc.request("mcp.ping", b"hello-from-gateway", timeout = 2.0)
        return {"mcp_reply": msg.data.decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}

import os, httpx

app  =  FastAPI()

@app.get('/healthz')
def health():
    return {'ok': True}

@app.get('/api/ping')
def ping():
    return {'ping': 'pong'}

@app.get('/api/llm/info')
async def llm_info():
    base  =  os.getenv('LLM_BASE_URL')
    if not base:
        return {'error': 'LLM_BASE_URL not configured'}
    try:
        async with httpx.AsyncClient(timeout = 5, verify = True) as cx:
            r  =  await cx.get(f'{base}/api/tags')
        return {'llm_base': base, 'status': r.status_code}
    except Exception as e:
        return {'llm_base': base, 'error': str(e)}
    try:
        async with httpx.AsyncClient(timeout = 5) as cx:
            r  =  await cx.get(f'{base}/api/tags')
        return {'llm_base': base, 'status': r.status_code}
    except Exception as e:
        return {'llm_base': base, 'error': str(e)}
     main
