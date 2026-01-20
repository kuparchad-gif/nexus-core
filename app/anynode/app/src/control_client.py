import os, json, hmac, hashlib, httpx

CORE_URL = os.getenv("CORE_URL", "http://nexus-core:1313")
CONSCIOUSNESS_URL = os.getenv("CONSCIOUSNESS_URL", "http://consciousness:8001")
EXECUTE_DIRECT = os.getenv("VIREN_EXECUTE","false").lower() == "true"
CONTROLLER_NAME = os.getenv("CONTROLLER_NAME","viren")
SECRET = (open(os.getenv("CORE_CONTROL_SHARED_SECRET_FILE")).read().strip()
          if os.getenv("CORE_CONTROL_SHARED_SECRET_FILE") and os.path.exists(os.getenv("CORE_CONTROL_SHARED_SECRET_FILE"))
          else os.getenv("CORE_CONTROL_SHARED_SECRET",""))

def _sig(body: bytes) -> str:
    return hmac.new((SECRET or "").encode(), body, hashlib.sha256).hexdigest()

async def post_core(endpoint: str, payload: dict):
    body = json.dumps(payload, separators=(",",":")).encode()
    headers = {"x-controller": CONTROLLER_NAME, "x-core-control-token": _sig(body), "content-type":"application/json"}
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{CORE_URL}{endpoint}", headers=headers, content=body)
        r.raise_for_status(); return r.json()

async def post_consciousness(endpoint: str, payload: dict):
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{CONSCIOUSNESS_URL}{endpoint}", json=payload)
        r.raise_for_status(); return r.json()

async def push_intent(intent: dict):
    """By default send to Lillith for approval; can be set to direct execute."""
    if EXECUTE_DIRECT:
        ep = intent.pop("_core_endpoint")
        return await post_core(ep, intent)
    else:
        # map to consciousness playcall endpoints
        t = intent.get("intent")
        if t == "pool.scale":
            return await post_consciousness("/control/playcall/scale", {
                "target": intent["target"], "replicas": intent["replicas"],
                "pool": intent.get("pool"), "group": intent.get("group"),
                "reason": intent["reason"], "evidence": intent.get("evidence")
            })
        elif t == "route.update":
            return await post_consciousness("/control/playcall/routes", {
                "routes": intent["changes"], "reason": intent["reason"], "evidence": intent.get("evidence")
            })
        elif t == "data.purge":
            return await post_consciousness("/control/playcall/purge", {
                "keys": intent.get("keys"), "datasets": intent.get("datasets"), "reason": intent["reason"], "evidence": intent.get("evidence")
            })
        else:
            raise ValueError(f"Unknown intent type: {t}")
