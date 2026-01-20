# discovery_kv/seeder.py
# [Path] C:\Projects\Stacks\nexus-metatron\policy\control_plane_kit\discovery_kv\seeder.py
import os, json, asyncio, sys
from pathlib import Path
from dotenv import load_dotenv
from nats.aio.client import Client as NATS
from nats.js.kv import KeyValue, KeyValueConfig

def load_required_caps(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["profiles"]

async def seed(nats_url: str, profiles: dict):
    nc = NATS()
    await nc.connect(servers=[nats_url])
    js = nc.jetstream()
    # Create KV bucket if not exists
    try:
        kv = await js.create_key_value(KeyValueConfig(bucket="DISCOVERY"))
    except Exception:
        kv = await js.key_value("DISCOVERY")

    for profile, spec in profiles.items():
        key = f"profiles.{profile}.required_caps"
        val = json.dumps(spec).encode("utf-8")
        await kv.put(key, val)
        print(f"Seeded {key}")

    await nc.drain()
    await nc.close()

def main():
    root = Path(__file__).parent.parent
    req_caps_path = root.parent / "discovery.required_caps.json"
    nats_url = os.getenv("NATS_SYS_URL") or os.getenv("NATS_HOME_URL") or "nats://127.0.0.1:4222"
    profiles = load_required_caps(str(req_caps_path))
    asyncio.run(seed(nats_url, profiles))

if __name__ == "__main__":
    main()
