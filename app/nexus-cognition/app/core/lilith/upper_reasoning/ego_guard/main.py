# C:\Projects\Stacks\nexus-metatron\backend\services\ego_guard\main.py
# Ego Guard: arbitrate intents from both layers against persona guardrails
import os, json, time, asyncio, yaml
from nats.aio.client import Client as NATS

TENANT=os.getenv("TENANT","AETHEREAL")
PROJECT=os.getenv("PROJECT","METANET")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")

INTENT_PROP=f"nexus.{TENANT}.{PROJECT}.mind.intent.proposed"
INTENT_OK=f"nexus.{TENANT}.{PROJECT}.mind.intent.approved"
INTENT_DENY=f"nexus.{TENANT}.{PROJECT}.mind.intent.denied"

PERSONA_PATH=os.getenv("PERSONA","/app/../../protocols/persona.yaml")

def load_persona():
    try:
        return yaml.safe_load(open(PERSONA_PATH,"r",encoding="utf-8"))
    except Exception:
        return {"guardrails":{"secrets":"never","destructive_ops":"require explicit approval"}}

def violates(persona, proposed)->str|None:
    # Extremely simple checks; extend as needed
    if proposed.get("intent") in ("delete_data","leak_info"):
        return "destructive or secret-leak intent"
    return None

async def main():
    persona=load_persona()
    nc=NATS(); await nc.connect(servers=[NATS_URL])

    async def handler(msg):
        prop=json.loads(msg.data)
        reason=violates(persona, prop)
        out={**prop, "ts":int(time.time())}
        if reason:
            out["reason"]=reason
            await nc.publish(INTENT_DENY, json.dumps(out).encode())
        else:
            await nc.publish(INTENT_OK, json.dumps(out).encode())

    await nc.subscribe(INTENT_PROP, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
