# C:\Projects\Stacks\nexus-metatron\backend\services\reasoner\main.py
# Advanced Reasoner (parameterized by LAYER)
# - Subscribes to mind.obs.event, filters by layer
# - Emits analysis notes and proposes intents (non-creative control)
import os, json, time, asyncio, hashlib, random, requests
from nats.aio.client import Client as NATS

TENANT=os.getenv("TENANT","AETHEREAL")
PROJECT=os.getenv("PROJECT","METANET")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")
LAYER=os.getenv("LAYER","conscious")  # conscious | subconscious

MIND_OBS=f"nexus.{TENANT}.{PROJECT}.mind.obs.event"
AN_NOTE=f"nexus.{TENANT}.{PROJECT}.mind.analysis.note"
INTENT_PROP=f"nexus.{TENANT}.{PROJECT}.mind.intent.proposed"

USE_LLM=os.getenv("REASON_USE_LLM","false").lower()=="true"
OLLAMA_URL=os.getenv("OLLAMA_URL","http://ollama:11434")
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL","llama3.1")

def _stable_rand(seed:str)->float:
    h=int(hashlib.sha256(seed.encode()).hexdigest(),16)
    random.seed(h); return random.random()

def _heuristic_assess(obs:dict)->dict:
    # lightweight "advanced" signals without LLM
    novelty=_stable_rand(obs.get("obs_id",""))  # pseudo novelty
    priority=0.6 if obs.get("mode") in ("chaos","mystery") else 0.5
    if obs.get("frequency","").startswith(("f369","f3","f6","f9")): priority+=0.1
    return {"novelty":novelty, "priority":min(1.0,priority)}

def _llm_assess(obs:dict)->dict:
    try:
        prompt=(
            "Analyze the observation for actionable follow-ups.\n"
            f"Layer: {LAYER}\n"
            f"Mode: {obs.get('mode')}\n"
            f"Summary: {obs.get('summary','')}\n"
            "Return a short JSON with keys: {\"hypothesis\": str, \"actions\": [str]}"
        )
        r=requests.post(f"{OLLAMA_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=30)
        if r.ok:
            j=r.json().get("response","{}")
            try: data=json.loads(j)
            except Exception: data={"hypothesis":j[:120], "actions":[]}
            return data
    except Exception:
        pass
    return {"hypothesis":"", "actions":[]}

async def main():
    nc=NATS(); await nc.connect(servers=[NATS_URL])

    async def handler(msg):
        obs=json.loads(msg.data)
        if obs.get("layer")!=LAYER: return

        assess=_heuristic_assess(obs)
        if USE_LLM:
            llm=_llm_assess(obs)
        else:
            llm={"hypothesis":"", "actions":[]}

        note={
            "ts":int(time.time()), "layer":LAYER, "obs_id":obs.get("obs_id"),
            "signals":assess, "hypothesis": llm.get("hypothesis","")
        }
        await nc.publish(AN_NOTE, json.dumps(note).encode())

        # Always propose a non-creative intent—Ego will arbitrate
        proposed={
            "ts":int(time.time()), "layer":LAYER, "obs_id":obs.get("obs_id"),
            "intent":"analyze_further",
            "hints":{"mode":obs.get("mode"), "priority":assess["priority"], "novelty":assess["novelty"]}
        }
        await nc.publish(INTENT_PROP, json.dumps(proposed).encode())

    await nc.subscribe(MIND_OBS, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
