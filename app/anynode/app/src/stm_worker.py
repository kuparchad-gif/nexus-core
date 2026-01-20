# stm_worker.py
# Short-Term Memory (STM) worker — CPU-parallel shard.
import asyncio, os, json, time, base64
from typing import Any, Dict
import xxhash
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout
from redis.asyncio import Redis

from . import bcu
from .crypto_box import seal as aead_seal, open as aead_open, load_key_from_env

LOG_LEVEL = os.getenv("LOG_LEVEL","INFO").upper()

def log(level, *args):
    levels = ["DEBUG","INFO","WARN","ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), level, *args, flush=True)

NATS_URL = os.getenv("NATS_URL","nats://nats:4222")
REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")
SHARD_INDEX = int(os.getenv("SHARD_INDEX","0"))
SHARD_COUNT = int(os.getenv("SHARD_COUNT","1"))
STM_TTL_MS = int(os.getenv("STM_TTL_MS","1800000"))
CRYPTO_CIPHER = os.getenv("CRYPTO_CIPHER","chacha20")
SEED_MAX_BUNDLE_BYTES = int(os.getenv("SEED_MAX_BUNDLE_BYTES","1048576"))
SEED_KEY_ROTATE_MIN = int(os.getenv("SEED_KEY_ROTATE_MIN","10"))
PLANNER_FANOUT_MAX = int(os.getenv("PLANNER_FANOUT_MAX","8"))
PLANNER_DEEP_CONTEXT_TAGS = [t.strip() for t in os.getenv("PLANNER_DEEP_CONTEXT_TAGS","").split(",") if t.strip()]

SEED_KEY = load_key_from_env("SEED_SHARED_KEY_BASE64")

def shard_for_id(_id: str) -> int:
    return xxhash.xxh32(_id).intdigest() % SHARD_COUNT

def subject_for(base: str) -> str:
    return f"{base}.{SHARD_INDEX}"

async def handle_put(redis: Redis, nc: NATS, msg):
    try:
        data = json.loads(msg.data.decode("utf-8"))
        _id = data["id"]
        if shard_for_id(_id) != SHARD_INDEX:
            # ignore; wrong shard
            return
        payload = data["payload"]
        use_bcu = bool(data.get("bcu"))
        ttl_ms = int(data.get("ttl_ms", STM_TTL_MS))

        blob = bcu.encode(payload) if use_bcu else json.dumps(payload).encode("utf-8")
        # store in redis with PX (ms) TTL
        await redis.set(f"stm:{_id}", blob, px=ttl_ms)
        await redis.set(f"stm:meta:{_id}", json.dumps({"tags": data.get("tags",[]), "hot": True}), px=ttl_ms)

        # respond if a reply subject exists
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"ok": True}).encode("utf-8"))
        # emit hotness event
        await nc.publish(f"mesh.think.events.stm.hotness.{SHARD_INDEX}".encode(), json.dumps({"id": _id, "hot": True}).encode())
    except Exception as e:
        log("ERROR","put failed", repr(e))

async def handle_get(redis: Redis, nc: NATS, msg):
    try:
        data = json.loads(msg.data.decode("utf-8"))
        _id = data["id"]
        if shard_for_id(_id) != SHARD_INDEX:
            return
        raw = await redis.get(f"stm:{_id}")
        meta = await redis.get(f"stm:meta:{_id}")
        if not raw:
            if msg.reply:
                await nc.publish(msg.reply, json.dumps({"ok": False, "error":"not_found"}).encode())
            return
        # try to decode as bcu; if fails, pass as utf-8 json
        try:
            val = bcu.decode(raw)
        except Exception:
            try:
                val = json.loads(raw.decode("utf-8"))
            except Exception:
                val = base64.b64encode(raw).decode()
        res = {"ok": True, "value": val, "meta": json.loads(meta) if meta else {}}
        if msg.reply:
            await nc.publish(msg.reply, json.dumps(res).encode("utf-8"))
    except Exception as e:
        log("ERROR","get failed", repr(e))

async def handle_seed(redis: Redis, nc: NATS, msg):
    try:
        data = json.loads(msg.data.decode("utf-8"))
        target = data.get("target","all")
        ids = data.get("ids",[])  # explicit ids
        # assemble bundle
        bundle_items = []
        total = 0
        for _id in ids:
            if shard_for_id(_id) != SHARD_INDEX:
                continue
            raw = await redis.get(f"stm:{_id}")
            if not raw: 
                continue
            entry = {"id": _id, "offset": 0, "len": len(raw)}
            total += len(raw)
            if total > SEED_MAX_BUNDLE_BYTES:
                break
            bundle_items.append(entry)
        bundle = {"bundle_id": f"{int(time.time()*1000)}-{SHARD_INDEX}", "items": bundle_items}
        blob = json.dumps(bundle).encode("utf-8")
        sealed = aead_seal(blob, SEED_KEY, CRYPTO_CIPHER)
        await nc.publish(f"mesh.think.events.stm.seeded.{target}".encode(), sealed)
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"ok": True, "bundle_bytes": len(sealed)}).encode())
    except Exception as e:
        log("ERROR","seed failed", repr(e))

async def main():
    log("INFO", f"STM worker starting shard={SHARD_INDEX}/{SHARD_COUNT}")
    redis = Redis.from_url(REDIS_URL, decode_responses=False)
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    # subscribe per-shard
    sub_put = await nc.subscribe(subject_for("mesh.think.cap.request.stm.put.1"), cb=lambda m: asyncio.create_task(handle_put(redis, nc, m)))
    sub_get = await nc.subscribe(subject_for("mesh.think.cap.request.stm.get.1"), cb=lambda m: asyncio.create_task(handle_get(redis, nc, m)))
    sub_seed = await nc.subscribe(subject_for("mesh.think.cap.request.stm.seed.1"), cb=lambda m: asyncio.create_task(handle_seed(redis, nc, m)))
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await nc.drain()
        await redis.close()

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
