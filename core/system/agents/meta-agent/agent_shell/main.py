# C:\Projects\Stacks\nexus-metatron\backend\services\agent_shell\main.py
# Agent Shell: front door; routes intents into the dual-layer mind honoring persona
import os, json, time, asyncio, yaml
from nats.aio.client import Client as NATS

TENANT=os.getenv("TENANT","AETHEREAL")
PROJECT=os.getenv("PROJECT","METANET")
NATS_URL=os.getenv("NATS_URL","nats://nats:4222")

AGENT_IN=f"nexus.{TENANT}.{PROJECT}.agent.intent"
AN_NOTE=f"nexus.{TENANT}.{PROJECT}.mind.analysis.note"
INTENT_OK=f"nexus.{TENANT}.{PROJECT}.mind.intent.approved"

PERSONA_PATH=os.getenv("PERSONA","/app/../../protocols/persona.yaml")

def load_persona():
    try: return yaml.safe_load(open(PERSONA_PATH,"r",encoding="utf-8"))
    except Exception: return {}

async def main():
    persona=load_persona()
    nc=NATS(); await nc.connect(servers=[NATS_URL])

    async def on_agent_intent(msg):
        intent=json.loads(msg.data)
        # For now: broadcast as interaction text, mind will take it from there
        text=intent.get("text") or intent.get("goal","")
        if text:
            topic=f"nexus.{TENANT}.{PROJECT}.interaction.text"
            await nc.publish(topic, json.dumps({"text":text}).encode())

    async def on_note(msg):
        # Could fuse into a response stream or UI; we just log now
        print(msg.data.decode())

    async def on_approved(msg):
        # In a real system: execute or route to planner; here: log
        print(msg.data.decode())

    await nc.subscribe(AGENT_IN, cb=on_agent_intent)
    await nc.subscribe(AN_NOTE, cb=on_note)
    await nc.subscribe(INTENT_OK, cb=on_approved)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
