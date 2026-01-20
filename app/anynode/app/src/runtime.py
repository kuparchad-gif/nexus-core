import os, asyncio, time, json, base64
from typing import Dict, Any

NATS_URL = os.getenv("NATS_URL", "nats://[REDACTED-IP]:4222")
LLM_URL  = os.getenv("LLM_PRIMARY_URL", "[REDACTED-URL])
MODEL    = os.getenv("NIM_MODEL", "qwen2.5:7b")

try:
    import nats
except Exception:
    nats = None

try:
    import httpx
except Exception:
    httpx = None

async def infer(prompt: str) -> str:
    if not httpx:
        return "[no-httpx] " + prompt[:128]
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{LLM_URL}/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=30)
        ct = r.headers.get("content-type","")
        return r.json().get("response","") if "application/json" in ct else r.text

async def chain(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt") or payload.get("text") or json.dumps(payload)[:2000]
    out = await infer(prompt)
    payload["inference"] = out
    payload["ts_done"] = int(time.time()*1000)
    return payload

async def main():
    if nats is None:
        print("nats-py missing")
        return
    nc = await nats.connect(servers=[NATS_URL])
    print("Connected:", NATS_URL)

    subjects = ["mem.shard.*.remember", "bin.nim.frames", "mesh.shard.register"]

    async def cb(msg):
        try:
            data = json.loads(msg.data.decode("utf-8"))
        except Exception:
            data = {"raw": base64.b64encode(msg.data).decode("ascii")}
        data["subject"] = msg.subject
        enriched = await chain(data)
        await nc.publish("mem.enriched.events", json.dumps(enriched).encode("utf-8"))

    subs = [await nc.subscribe(s, cb=cb) for s in subjects]
    print("Subscribed:", subjects)
    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        for s in subs: await s.unsubscribe()
        await nc.drain()

if __name__ == "__main__":
    asyncio.run(main())

