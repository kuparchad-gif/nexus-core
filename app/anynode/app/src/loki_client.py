import os, time, json, requests
LOKI_URL = (os.environ.get("LOKI_URL") or "[REDACTED-URL]).rstrip("/")
ORG = os.environ.get("LOKI_ORGID") or "fake"
def _ns_now(): return str(int(time.time() * 1_000_000_000))
def push_log(level="INFO", message="ok", labels=None):
    try:
        labels = labels or {}
        stream = {"job":"viren","level":level}
        stream.update(labels)
        payload = {"streams":[{"stream":stream,"values":[[_ns_now(), message]]}]}
        headers = {"Content-Type":"application/json"}
        if ORG: headers["X-Scope-OrgID"] = ORG
        r = requests.post(f"{LOKI_URL}/loki/api/v1/push", headers=headers, data=json.dumps(payload), timeout=3)
        r.raise_for_status()
    except Exception:
        pass

