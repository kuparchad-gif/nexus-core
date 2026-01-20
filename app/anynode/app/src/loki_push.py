import os, json, time
try:
    import httpx
except Exception:
    httpx = None

LOKI_URL = os.getenv("LOKI_URL", "[REDACTED-URL])

def push_log(stream: str, msg: str, labels: dict = None):
    if not httpx:
        return
    labels = labels or {}
    payload = {
        "streams": [{
            "stream": {"job": stream, **labels},
            "values": [[str(int(time.time()*1e9)), msg]]
        }]
    }
    try:
        httpx.post(LOKI_URL, json=payload, timeout=2.0)
    except Exception:
        pass

