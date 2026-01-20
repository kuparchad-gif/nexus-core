from fastapi import FastAPI
import os, asyncio, httpx, json, time

app = FastAPI(title="Archiver BookWorm", version="0.1.0")

ARCHIVER_URL = os.getenv("ARCHIVER_URL","http://localhost:9020")
WORM_SOURCES = os.getenv("WORM_SOURCES_JSON","[]")  # '["http://loki:9011","http://viren:9012"]'
WORM_INTERVAL_S = int(os.getenv("WORM_INTERVAL_S","60"))
EXPORT_PATH = os.getenv("WORM_EXPORT_PATH","/db/export")
SCRUB_PATH  = os.getenv("WORM_SCRUB_PATH","/db/scrub")

@app.get("/health")
def health():
    return {"status":"ok","archiver":ARCHIVER_URL,"interval":WORM_INTERVAL_S,"sources":WORM_SOURCES}

async def run_once():
    try:
        sources = json.loads(WORM_SOURCES)
    except Exception:
        sources = []
    payload = {"sources":[s for s in sources], "export_path": EXPORT_PATH, "scrub_path": SCRUB_PATH}
    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            r = await c.post(ARCHIVER_URL.rstrip("/") + "/worms/run", json=payload)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

@app.on_event("startup")
async def _loop():
    async def loop():
        while True:
            await run_once()
            await asyncio.sleep(WORM_INTERVAL_S)
    asyncio.create_task(loop())
