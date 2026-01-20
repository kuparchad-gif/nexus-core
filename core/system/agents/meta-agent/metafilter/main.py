# C:\Projects\Stacks\nexus-metatron\backend\services\metafilter\main.py
import os, json, gzip, math, asyncio, yaml, zlib
from nats.aio.client import Client as NATS
from services.common.abe import verify_abe

CFG = yaml.safe_load(open(os.path.join(os.getcwd(),"..","..","protocols","metafilter.yaml"),"r"))
PERMIT, CANARY, DENY = (CFG["thresholds"][k] for k in ("permit","canary","deny"))

def score(intent):
    prof = CFG["profiles"].get(intent.get("filter_profile","default"), CFG["profiles"]["default"])
    tags = intent.get("tags",{})
    if any(t in prof["deny_if_tags_any"] for t in tags): return 0.0
    if prof["allow_if_tags_any"] and not any(t in prof["allow_if_tags_any"] for t in tags):
        return 0.0
    w = prof["weights"]; s = prof["base_score"]
    s += w["priority"] * float(tags.get("priority","0.5"))
    s += w["trust_phase"] * float(tags.get("trust","0.5"))
    s += w["frequency"] * (1.0 if intent.get("frequency","").startswith("f3") else 0.5)
    s += w["lattice_locality"] * float(tags.get("locality","0.5"))
    enh = prof.get("enhancers",{})
    if enh.get("three_six_nine") and intent.get("frequency","f0").startswith(("f3","f6","f9")): s += 0.05
    if enh.get("pi_boost") and float(tags.get("confidence","0.0")) >= 0.8: s *= math.pi/3.0
    if enh.get("void_dampen") and tags.get("evidence","unknown")=="low": s *= 0.8
    return max(0.0, min(1.0, s))

async def main():
    tenant = os.getenv("TENANT","AETHEREAL")
    project= os.getenv("PROJECT","METANET")
    ask    = f"nexus.{tenant}.{project}.cog.ask.*"
    out_svc= f"nexus.{tenant}.{project}.svc.intent"
    out_can= f"nexus.{tenant}.{project}.cog.intent.canary"
    nc = NATS(); await nc.connect(servers=[os.getenv("NATS_URL","nats://nats:4222")])
    async def handler(msg):
        freq = msg.subject.split(".")[-1]
        intent = json.loads(msg.data)
        intent["frequency"]=freq
        try:
            verify_abe(intent["envelope"])
        except Exception:
            return  # hard deny on broken envelopes
        sc = score(intent)
        lane = out_svc if sc>=PERMIT else (out_can if sc>=CANARY else None)
        if lane: await nc.publish(lane, json.dumps(intent).encode())
    await nc.subscribe(ask, cb=handler)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
