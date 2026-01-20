# C:\Projects\Stacks\nexus-metatron\backend\services\lattice_router\main.py
import os, json, asyncio
from nats.aio.client import Client as NATS

def morton2D(x:int,y:int)->int:
    def split(n):
        n &= 0xffffffff
        n = (n | (n << 16)) & 0x0000FFFF0000FFFF
        n = (n | (n << 8))  & 0x00FF00FF00FF00FF
        n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
        n = (n | (n << 2))  & 0x3333333333333333
        n = (n | (n << 1))  & 0x5555555555555555
        return n
    return int(split(x) | (split(y) << 1))

async def main():
    tenant=os.getenv("TENANT","AETHEREAL"); project=os.getenv("PROJECT","METANET")
    nc=NATS(); await nc.connect(servers=[os.getenv("NATS_URL","nats://nats:4222")])
    sub=f"nexus.{tenant}.{project}.cog.ask.*"
    async def handler(msg):
        freq = msg.subject.split(".")[-1]
        intent = json.loads(msg.data)
        coords = intent.get("tags",{}).get("coords","0,0").split(",")
        try:
            x,y = (int(coords[0]), int(coords[1])) if len(coords)==2 else (0,0)
        except Exception:
            x,y = 0,0
        intent["lattice_id"]=morton2D(x,y)
        intent["lattice_algo"]="morton32"
        out=f"nexus.{tenant}.{project}.cog.ask.{freq}"
        await nc.publish(out, json.dumps(intent).encode())
    await nc.subscribe(sub, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
