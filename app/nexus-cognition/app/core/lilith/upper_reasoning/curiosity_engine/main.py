# C:\Projects\Stacks\nexus-metatron\backend\services\curiosity_engine\main.py
# Curiosity Engine (Observatory-only)
# - Subscribes to cog.result.*
# - Emits normalized observation events to cog.obs.event (no generative content)
# - Optionally signals "ask how" as a boolean flag on cog.how.ask (no prose)
# - Persists a ledger record to Qdrant (payload-first)
import os, json, time, asyncio, gzip, uuid
from typing import Tuple, Any, Dict

from nats.aio.client import Client as NATS
from services.common.abe import verify_abe

# Optional: Qdrant persistence
QDRANT_ENABLE = os.getenv("QDRANT_ENABLE","true").lower() == "true"
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except Exception:
    QDRANT_ENABLE = False
    QdrantClient = None

TENANT  = os.getenv("TENANT","AETHEREAL")
PROJECT = os.getenv("PROJECT","METANET")
NATS_URL= os.getenv("NATS_URL","nats://nats:4222")

RESULT_SUB = f"nexus.{TENANT}.{PROJECT}.cog.result.*"
OBS_EVENT  = f"nexus.{TENANT}.{PROJECT}.cog.obs.event"    # ← normalized observations
HOW_ASK    = f"nexus.{TENANT}.{PROJECT}.cog.how.ask"      # ← boolean signal only
HOW_NOTE   = f"nexus.{TENANT}.{PROJECT}.cog.how.note"

LEDGER_COLLECTION = os.getenv("LEDGER_COLLECTION","curiosity_ledger")
QDRANT_URL = os.getenv("QDRANT_URL","http://qdrant:6333")

# ---- helpers ----
def _try_decode_abe(envelope: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    try:
        verify_abe(envelope)  # CRC + optional sig
        header = json.loads(gzip.decompress(envelope["header"]).decode())
        payload = gzip.decompress(envelope["payload"])
        return header, payload
    except Exception:
        return {}, b""

def _should_ask_how(result: Dict[str, Any], freq: str, header: Dict[str, Any]) -> bool:
    ok = result.get("ok", False)
    if not ok:
        return False
    if freq.startswith(("f369","f3","f6","f9")):
        return True
    quality = float(header.get("metrics",{}).get("quality", 0.0))
    return quality >= 0.80

def _make_observation(result: Dict[str, Any], freq: str, header: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic, non-creative observation schema
    return {
        "obs_id": uuid.uuid4().hex,
        "ts": int(time.time()),
        "tenant": TENANT,
        "project": PROJECT,
        "frequency": freq,
        "task_id": result.get("id") or result.get("ask_id") or "",
        "ok": bool(result.get("ok", False)),
        "summary": result.get("summary",""),
        "error": result.get("error",""),
        "metrics": header.get("metrics", {}),
        "tags": result.get("tags", {}),
        "mode": result.get("tags",{}).get("mode","curiosity"),
        # zero creativity: no generated prose, only structured fields
    }

async def _ensure_qdrant(client: QdrantClient):
    try:
        client.get_collection(LEDGER_COLLECTION)
    except Exception:
        client.recreate_collection(
            collection_name=LEDGER_COLLECTION,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

async def run():
    nc = NATS(); await nc.connect(servers=[NATS_URL])

    qdr = None
    if QDRANT_ENABLE and QdrantClient:
        qdr = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        await _ensure_qdrant(qdr)

    async def on_result(msg):
        freq = msg.subject.split(".")[-1]
        try:
            result = json.loads(msg.data)
        except Exception:
            return
        header, payload = _try_decode_abe(result.get("envelope", {}))
        obs = _make_observation(result, freq, header)
        await nc.publish(OBS_EVENT, json.dumps(obs).encode())

        # boolean "ask-how" signal only (no creative content)
        if _should_ask_how(result, freq, header):
            signal = {
                "ts": int(time.time()),
                "task_id": obs["task_id"],
                "frequency": freq,
                "ask_how": True
            }
            await nc.publish(HOW_ASK, json.dumps(signal).encode())

        # Persist a simple ledger copy for traceability
        if qdr:
            try:
                from qdrant_client.models import PointStruct
                pt = PointStruct(id=obs["obs_id"], vector=[0.0,0.0,0.0,0.0], payload=obs)
                qdr.upsert(collection_name=LEDGER_COLLECTION, points=[pt])
            except Exception as e:
                print(json.dumps({"event":"curiosity_qdrant_error","error":str(e)}))

    await nc.subscribe(RESULT_SUB, cb=on_result)

    # Keep how.notes purely logged (no transformation)
    async def on_note(msg):
        try: note = json.loads(msg.data)
        except Exception: return
        print(json.dumps({"event":"curiosity_note","note":note}))

    await nc.subscribe(HOW_NOTE, cb=on_note)

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(run())
