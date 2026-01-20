# C:\Projects\Stacks\nexus-metatron\backend\services\planner\main.py
import os, json, asyncio, uuid, time
from nats.aio.client import Client as NATS
from services.common.abe import pack_abe

async def main():
    tenant=os.getenv("TENANT","AETHEREAL"); project=os.getenv("PROJECT","METANET")
    freq=os.getenv("DEFAULT_FREQ","f369")
    ask=f"nexus.{tenant}.{project}.cog.ask.{freq}"
    nc=NATS(); await nc.connect(servers=[os.getenv("NATS_URL","nats://nats:4222")])
    # demo tick: emit a heartbeat intent every 10s
    while True:
        header={"freq":freq,"algo":"abe-v1","ts":int(time.time())}
        payload=b'{"ping":"hello"}'
        intent={
            "id": uuid.uuid4().hex,
            "tier": 2,
            "kind": 0,
            "subject": "demo-intent",
            "goal": "say hello",
            "frequency": freq,
            "lattice_id": 0,
            "lattice_algo": "morton32",
            "filter_profile": "default",
            "envelope": pack_abe(header, payload),
            "tags": {"priority":"0.5","trust":"0.6","locality":"0.5","confidence":"0.9" },
            "version": "v1.0.0"
        }
        await nc.publish(ask, json.dumps(intent).encode())
        await asyncio.sleep(10)

if __name__=="__main__":
    asyncio.run(main())
