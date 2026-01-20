# services/reasoner/main.patched.py
# Drop this in place of backend/services/reasoner/main.py in your pack.
import os, json, time, asyncio, hashlib, random, requests
from nats.aio.client import Client as NATS

TENANT=os.getenv("TENANT","AETHEREAL")
PROJECT=os.getenv("PROJECT","METANET")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")
LAYER=os.getenv("LAYER","conscious")

MIND_OBS=f"nexus.{TENANT}.{PROJECT}.mind.obs.event"
AN_NOTE=f"nexus.{TENANT}.{PROJECT}.mind.analysis.note"
CB_ASK=f"nexus.{TENANT}.{PROJECT}.cogbridge.ask.reason"

def _stable_rand(seed:str)->float:
    h=int(hashlib.sha256(seed.encode()).hexdigest(),16)
    random.seed(h); return random.random()

def _heuristic_assess(obs:dict)->dict:
    novelty=_stable_rand(obs.get("obs_id",""))
    priority=0.6 if obs.get("mode") in ("chaos","mystery") else 0.5
    if obs.get("frequency","").startswith(("f369","f3","f6","f9")): priority+=0.1
    return {"novelty":novelty, "priority":min(1.0,priority)}

async def main():
    nc=NATS(); await nc.connect(servers=[NATS_URL])
    async def handler(msg):
        obs=json.loads(msg.data)
        if obs.get("layer")!=LAYER: return
        assess=_heuristic_assess(obs)
        note={"ts":int(time.time()),"layer":LAYER,"obs_id":obs.get("obs_id"),"signals":assess}
        await nc.publish(AN_NOTE, json.dumps(note).encode())

        ask_payload={
            "layer": LAYER,
            "observation": {"obs_id": obs.get("obs_id"), "summary": obs.get("summary",""), "mode": obs.get("mode","curiosity")},
            "profile": None,
            "goal": "Analyze and advise next steps.",
            "hints": {"priority": assess["priority"], "novelty": assess["novelty"]}
        }
        await nc.publish(CB_ASK, json.dumps(ask_payload).encode())
    await nc.subscribe(MIND_OBS, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())

