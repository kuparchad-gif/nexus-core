# cognikubes/nexus_core/services/browser_receiver/app/nexus_envelope.py
import os, time, uuid
from typing import Dict, Any

def build_envelope(payload: Dict[str, Any], req_id: str | None = None) -> Dict[str, Any]:
    rid = req_id or str(uuid.uuid4())
    labels = {
        "tenant": os.getenv("TENANT", "public"),
        "project": os.getenv("PROJECT", "nexus"),
        "service": os.getenv("SERVICE_NAME", "browser-receiver"),
        "topic": "chat",
        "privacy": "external",
    }
    return {
        "envelope": {"version": "nexus.v1", "id": rid, "ts": time.time(), "labels": labels},
        **payload,
    }

