import os, ssl, json, uuid, yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio, nats

CFG = os.getenv("WORMS_FILE","/worms/worms.yaml")
BUS = os.getenv("MCP_NATS_URL","nats://host.containers.internal:4222")

def tls_ctx():
    ctx = ssl.create_default_context(cafile="/certs/ca/ca.crt")
    ctx.load_cert_chain("/certs/client/client.crt","/certs/client/client.key")
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

async def main():
    with open(CFG,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    worms = cfg.get("worms", [])
    nc = await nats.connect(servers=[BUS], tls=tls_ctx())
    sched = AsyncIOScheduler(timezone="UTC")

    async def fire(subject, payload):
        job_id = uuid.uuid4().hex
        payload = dict(payload); payload["job_id"] = job_id
        try:
            msg = await nc.request(subject, json.dumps(payload).encode(), timeout=15.0)
            await nc.publish("logs.worms", json.dumps({"done":True, "job":job_id, "reply":msg.data.decode()}).encode())
        except Exception as e:
            await nc.publish("logs.worms", json.dumps({"error":str(e), "job":job_id}).encode())

    def add_job(cron, subject, payload):
        parts = cron.split()
        kw = dict(minute=parts[0], hour=parts[1], day=parts[2], month=parts[3], day_of_week=parts[4])
        sched.add_job(lambda: asyncio.create_task(fire(subject, payload)), trigger="cron", **kw)

    for w in worms: add_job(w["cron"], w["subject"], w.get("payload", {}))
    sched.start()
    print("wormkeeper online with", len(worms), "worms", flush=True)
    while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
