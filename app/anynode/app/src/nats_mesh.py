# service/nats_mesh.py (v2 — terminology aligned)
import os, json, asyncio, time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

import nats
from nats.js import api as jsapi

def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None else default

# NATS URLs — local + per-mesh (Meshes == Buses)
NATS_LOCAL_URL = env("NATS_LOCAL_URL", "nats://127.0.0.1:4222")
NATS_SENSE_URL = env("NATS_SENSE_URL", "nats://127.0.0.1:4222")
NATS_THINK_URL = env("NATS_THINK_URL", "nats://127.0.0.1:4222")
NATS_HEAL_URL  = env("NATS_HEAL_URL",  "nats://127.0.0.1:4222")
NATS_ARCH_URL  = env("NATS_ARCH_URL",  "nats://127.0.0.1:4222")

MESH_TO_URL = {
    "local": NATS_LOCAL_URL,
    "sense": NATS_SENSE_URL,
    "think": NATS_THINK_URL,
    "heal":  NATS_HEAL_URL,
    "archive": NATS_ARCH_URL,
}

STREAMS = {
    "sense":   ("JS_SENSE",   ["mesh.sense.>"]),
    "think":   ("JS_THINK",   ["mesh.think.>"]),
    "heal":    ("JS_HEAL",    ["mesh.heal.>"]),
    "archive": ("JS_ARCHIVE", ["mesh.archive.>"]),
}

async def ensure_stream(nc: nats.NATS, js, name: str, subjects: List[str]):
    try:
        await js.add_stream(
            name=name,
            subjects=subjects,
            retention=jsapi.RetentionPolicy.Limits,
            storage=jsapi.StorageType.File,
            num_replicas=1,
            max_msgs=-1,
            max_bytes=-1,
            max_age=0,
            discard=jsapi.DiscardPolicy.Old,
            allow_direct=True,
            deny_delete=False,
            duplicate_window=0
        )
    except Exception:
        try:
            si = await js.stream_info(name)
            cfg = si.config
            cfg.subjects = list(set(subjects))
            await js.update_stream(cfg)
        except Exception as e:
            print(f"[ensure_stream] {name}: {e}")

async def provision_mesh(meshes: List[str]) -> Dict[str, Any]:
    results = {}
    for mesh in meshes:
        url = MESH_TO_URL.get(mesh)
        name, subjects = STREAMS.get(mesh, (None, None))
        if not url or not name:
            results[mesh] = {"ok": False, "error": "unknown mesh/url"}
            continue
        try:
            nc = await nats.connect(servers=[url])
            js = nc.jetstream()
            await ensure_stream(nc, js, name, subjects)
            await nc.close()
            results[mesh] = {"ok": True, "url": url, "stream": name, "subjects": subjects}
        except Exception as e:
            results[mesh] = {"ok": False, "error": str(e), "url": url}
    return results

async def publish_capabilities(plan: Dict[str, Any]) -> Dict[str, Any]:
    results = []
    for m in plan.get("modules", []):
        mesh = m.get("mesh")
        url = MESH_TO_URL.get(mesh)
        subj = f"mesh.{mesh}.cap.register"
        payload = {
            "module": m.get("name"),
            "caps": m.get("caps", []),
            "mesh": mesh,
            "host": m.get("host", "unknown"),
            "endpoints": m.get("endpoints", {}),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        try:
            nc = await nats.connect(servers=[url])
            await nc.publish(subj, json.dumps(payload).encode("utf-8"))
            await nc.flush()
            await nc.close()
            results.append({"module": m.get("name"), "ok": True, "mesh": mesh, "url": url})
        except Exception as e:
            results.append({"module": m.get("name"), "ok": False, "error": str(e)})
    return {"results": results}
