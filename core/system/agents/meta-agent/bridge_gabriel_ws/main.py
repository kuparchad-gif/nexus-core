# C:\Projects\Stacks\nexus-metatron\backend\services\bridge_gabriel_ws\main.py
import os, json, asyncio, websockets
from nats.aio.client import Client as NATS

WS_URL   = os.getenv("GABRIEL_WS","ws://localhost:8765")
TENANT   = os.getenv("TENANT","AETHEREAL")
PROJECT  = os.getenv("PROJECT","METANET")
IN_SUB   = f"nexus.{TENANT}.{PROJECT}.gabriel.emit"      # NATS -> WS
OUT_SUB  = f"nexus.{TENANT}.{PROJECT}.gabriel.broadcast" # WS -> NATS

async def ws_to_nats(nc):
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        data = {"raw": msg}
                    await nc.publish(OUT_SUB, json.dumps(data).encode())
        except Exception:
            await asyncio.sleep(1.0)  # backoff

async def nats_to_ws(nc):
    async def handler(msg):
        payload = msg.data
        try:
            payload = json.dumps(json.loads(payload)).encode()
        except Exception:
            pass
        try:
            async with websockets.connect(WS_URL) as ws:
                await ws.send(payload.decode())
        except Exception:
            # drop if WS not available
            pass
    await nc.subscribe(IN_SUB, cb=handler)

async def main():
    nc = NATS(); await nc.connect(servers=[os.getenv("NATS_URL","nats://nats:4222")])
    await asyncio.gather(ws_to_nats(nc), nats_to_ws(nc))

if __name__=="__main__":
    asyncio.run(main())
