# C:\Projects\Stacks\nexus-metatron\backend\services\perception_engine\main.py
# Perception Engine (Observational)
# - Subscribes to interaction text
# - Classifies affect (valence/arousal/dominance) and initial mode per rules
# - Emits normalized observation to cog.obs.event
# - Emits a reflex plan to cog.reflex.plan (sequence of modes to follow)
import os, json, time, asyncio, uuid, re, yaml
from typing import Dict, Any

from nats.aio.client import Client as NATS

TENANT   = os.getenv("TENANT","AETHEREAL")
PROJECT  = os.getenv("PROJECT","METANET")
NATS_URL = os.getenv("NATS_URL","nats://nats:4222")

INTERACTION_IN = f"nexus.{TENANT}.{PROJECT}.interaction.text"
OBS_EVENT      = f"nexus.{TENANT}.{PROJECT}.cog.obs.event"
REFLEX_PLAN    = f"nexus.{TENANT}.{PROJECT}.cog.reflex.plan"

RULES_PATH = os.getenv("AFFECT_RULES","/app/../../protocols/affect_rules.yaml")

def load_rules()->Dict[str,Any]:
    try:
        return yaml.safe_load(open(RULES_PATH,"r",encoding="utf-8"))
    except Exception as e:
        return {"defaults":{"valence":0.5,"arousal":0.5,"dominance":0.5,"mode":"curiosity","sequence":["curiosity","abstract"]}, "rules":[]}

def classify(text:str, rules)->Dict[str,Any]:
    d = rules.get("defaults",{}).copy()
    for r in rules.get("rules",[]):
        hit = False
        if r.get("when_any"):
            lc = text.lower()
            for token in r["when_any"]:
                if token.lower() in lc:
                    hit = True; break
        if not hit and r.get("when_regex"):
            try:
                if re.search(r["when_regex"], text, flags=re.IGNORECASE|re.MULTILINE):
                    hit = True
            except Exception:
                pass
        if hit:
            d.update(r.get("set",{}))
            d["rule"] = r.get("name","")
            break
    return d

async def main():
    rules = load_rules()
    nc = NATS(); await nc.connect(servers=[NATS_URL])

    async def on_text(msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"text": msg.data.decode(errors="ignore")}
        text = payload.get("text","").strip()
        if not text:
            return
        aff = classify(text, rules)
        obs = {
            "obs_id": uuid.uuid4().hex,
            "ts": int(time.time()),
            "tenant": TENANT, "project": PROJECT,
            "source": "interaction",
            "text": text,
            "frequency": "f369",   # default cognitive lane
            "mode": aff.get("mode","curiosity"),
            "affect": {
                "valence": float(aff.get("valence",0.5)),
                "arousal": float(aff.get("arousal",0.5)),
                "dominance": float(aff.get("dominance",0.5)),
            },
            "tags": {"rule": aff.get("rule","")}
        }
        await nc.publish(OBS_EVENT, json.dumps(obs).encode())

        plan = {
            "obs_id": obs["obs_id"],
            "ts": obs["ts"],
            "tenant": TENANT, "project": PROJECT,
            "sequence": aff.get("sequence", ["curiosity","abstract"]),
            "initial_mode": obs["mode"]
        }
        await nc.publish(REFLEX_PLAN, json.dumps(plan).encode())

    await nc.subscribe(INTERACTION_IN, cb=on_text)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
