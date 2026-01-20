# C:\Projects\Stacks\nexus-metatron\backend\services\dream_engine\main.py
# Dream Engine: subconscious replay/simulation (creative allowed in sandbox)
import os, json, time, asyncio, random, hashlib, requests
from nats.aio.client import Client as NATS

TENANT=os.getenv("TENANT","AETHEREAL")
PROJECT=os.getenv("PROJECT","METANET")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")

MIND_OBS=f"nexus.{TENANT}.{PROJECT}.mind.obs.event"
SANDBOX=f"nexus.{TENANT}.{PROJECT}.cog.sandbox.intent"

USE_LLM=os.getenv("DREAM_USE_LLM","false").lower()=="true"
OLLAMA_URL=os.getenv("OLLAMA_URL","http://ollama:11434")
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL","llama3.1")

def _seed(s): random.seed(int(hashlib.sha256(s.encode()).hexdigest(),16)%2**32)

async def main():
    nc=NATS(); await nc.connect(servers=[NATS_URL])

    async def handler(msg):
        obs=json.loads(msg.data)
        if obs.get("layer")!="subconscious": return
        _seed(obs.get("obs_id",""))
        # Build a harmless sandbox replay intent
        sim={
            "ts": int(time.time()),
            "obs_id": obs.get("obs_id"),
            "mode": obs.get("mode"),
            "frequency": obs.get("frequency","f369"),
            "sim_depth": random.choice([1,2,3]),
            "note": "subconscious replay",
        }
        await nc.publish(SANDBOX, json.dumps({"ticket":{"type":"dream_replay","payload":sim}}).encode())

    await nc.subscribe(MIND_OBS, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
