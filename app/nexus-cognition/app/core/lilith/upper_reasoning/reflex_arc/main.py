# C:\Projects\Stacks\nexus-metatron\backend\services\reflex_arc\main.py
# Reflex Arc (Observational Sequencer)
# - Subscribes to cog.reflex.plan (sequence of modes)
# - Emits minimal control tickets to HOW/ASK or sandbox lanes (no creativity)
# - The Planner can consume these tickets to route analysis tasks
import os, json, time, asyncio

from nats.aio.client import Client as NATS

TENANT   = os.getenv("TENANT","AETHEREAL")
PROJECT  = os.getenv("PROJECT","METANET")
NATS_URL = os.getenv("NATS_URL","nats://nats:4222")

REFLEX_PLAN = f"nexus.{TENANT}.{PROJECT}.cog.reflex.plan"
HOW_ASK     = f"nexus.{TENANT}.{PROJECT}.cog.how.ask"
SANDBOX_INTENT = f"nexus.{TENANT}.{PROJECT}.cog.sandbox.intent"

def _tickets(obs_id:str, seq:list[str]):
    # Map modes into non-creative tickets; Planner decides the concrete work
    for m in seq:
        if m == "curiosity":
            yield {"type":"ask_how", "obs_id": obs_id, "reason":"reflex"}
        elif m == "chaos":
            yield {"type":"sandbox_probe", "obs_id": obs_id, "profile":"chaos-lite"}
        elif m == "mystery":
            yield {"type":"investigate_unknowns", "obs_id": obs_id}
        elif m == "wonder":
            yield {"type":"seek_surprise", "obs_id": obs_id}
        elif m == "imagination":
            yield {"type":"hypothesis_slot", "obs_id": obs_id}
        elif m == "abstract":
            yield {"type":"compress_pattern", "obs_id": obs_id}

async def main():
    nc = NATS(); await nc.connect(servers=[NATS_URL])

    async def on_plan(msg):
        try:
            plan = json.loads(msg.data)
        except Exception:
            return
        obs_id = plan.get("obs_id","")
        seq = plan.get("sequence",[])
        for t in _tickets(obs_id, seq):
            if t["type"] == "ask_how":
                await nc.publish(HOW_ASK, json.dumps({"ts":int(time.time()), "obs_id":obs_id, "ask_how":True}).encode())
            else:
                # all others go to sandbox intent as control hints (Planner-defined)
                await nc.publish(SANDBOX_INTENT, json.dumps({"ts":int(time.time()), "obs_id":obs_id, "ticket":t}).encode())

    await nc.subscribe(REFLEX_PLAN, cb=on_plan)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
