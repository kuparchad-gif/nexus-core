import os, time, httpx

ARCHIVER_URL = os.getenv("ARCHIVER_URL", "http://localhost:9020")
def emit_to_archiver(source: str, topic: str, payload: dict, tags=None, severity="INFO", phase=None, policy_level=None):
    body = {
        "source": source, "topic": topic, "payload": payload,
        "tags": tags or [], "severity": severity, "phase": phase, "policy_level": policy_level
    }
    try:
        with httpx.Client(timeout=5.0) as s:
            s.post(f"{ARCHIVER_URL}/archive", json=body)
    except Exception:
        pass
