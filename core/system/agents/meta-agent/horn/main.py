# C:\Projects\Stacks\nexus-metatron\backend\services\horn\main.py
import os, json, asyncio
import toml
from nats.aio.client import Client as NATS

CFG = toml.load(os.path.join(os.getcwd(),"..","..","protocols","gabriel.toml"))
FREG = json.load(open(os.path.join(os.getcwd(),"..","..","protocols","frequency_registry.json")))

def harmonics(freq):
    return FREG["lanes"].get(freq,{}).get("harmonics",[])

async def main():
    tenant=os.getenv("TENANT","AETHEREAL"); project=os.getenv("PROJECT","METANET")
    nc=NATS(); await nc.connect(servers=[os.getenv("NATS_URL","nats://nats:4222")])
    base=f"nexus.{tenant}.{project}.cog.ask.*"
    async def handler(msg):
        freq=msg.subject.split(".")[-1]
        if not CFG["trumpet"]["enable_harmonics"]: return
        cap=int(CFG["trumpet"]["fanout_cap"])
        fans=0
        for h in harmonics(freq):
            if fans>=cap: break
            subj=".".join(msg.subject.split(".")[:-1]+[h])
            await nc.publish(subj, msg.data); fans+=1
    await nc.subscribe(base, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
