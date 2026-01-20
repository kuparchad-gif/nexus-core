# C:\Projects\Stacks\nexus-metatron\backend\services\corpus_callosum\main.py
# Corpus Callosum
# - Subscribes to base cog.obs.event (layer-agnostic)
# - Emits two annotated observations into mind.obs.event with layer=conscious/subconscious
import os, json, time, asyncio
from nats.aio.client import Client as NATS

TENANT=os.getenv("TENANT","AETHEREAL")
PROJECT=os.getenv("PROJECT","METANET")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")

BASE_OBS=f"nexus.{TENANT}.{PROJECT}.cog.obs.event"
MIND_OBS=f"nexus.{TENANT}.{PROJECT}.mind.obs.event"
CALLOSUM_TAP=f"nexus.{TENANT}.{PROJECT}.mind.callosum.tap"

async def main():
    nc=NATS(); await nc.connect(servers=[NATS_URL])

    async def handler(msg):
        try:
            obs=json.loads(msg.data)
        except Exception:
            return
        for layer in ("conscious","subconscious"):
            o=dict(obs); o["layer"]=layer; o["ts_callosum"]=int(time.time())
            await nc.publish(MIND_OBS, json.dumps(o).encode())
        # Tap for debugging/metrics
        await nc.publish(CALLOSUM_TAP, msg.data)

    await nc.subscribe(BASE_OBS, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
