# Drop-in replacement for services/nova/app.py adding capabilities, offline mode, and canonicalization.
import os, json, re, hmac, hashlib
from typing import Dict, Any
from fastapi import FastAPI, Body

NOVA_VERSION = os.getenv("NOVA_VERSION","1.0.0")
CANON_ID = os.getenv("NOVA_CANON_ID","canon:stable@1")
OFFLINE = os.getenv("NOVA_OFFLINE","1") == "1"  # default offline

app = FastAPI(title="nova", version=NOVA_VERSION)

COSIGN_SECRET = ""
if os.getenv("CORE_NOVA_COSIGN_SECRET_FILE") and os.path.exists(os.getenv("CORE_NOVA_COSIGN_SECRET_FILE")):
    COSIGN_SECRET = open(os.getenv("CORE_NOVA_COSIGN_SECRET_FILE"), "r").read().strip()
else:
    COSIGN_SECRET = os.getenv("CORE_NOVA_COSIGN_SECRET","")

HUMAN_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ,.';:()\-]{6,}")

def canonicalize(payload: Dict[str, Any]) -> bytes:
    """Stable canonicalization: drop unknown meta fields, sort keys, no whitespace."""
    # Never include local meta in signature
    p = {k: v for k, v in payload.items() if not k.startswith("_")}
    return json.dumps(p, separators=(",",":"), sort_keys=True).encode()

@app.get("/meta/capabilities")
async def caps():
    return {
        "version": NOVA_VERSION,
        "canon_id": CANON_ID,
        "offline": OFFLINE,
        "lint_rules": ["reason_human","replica_bound","known_intent"],
        "cosign_algo": "hmac_sha256_body_canon_sortkeys"
    }

@app.get("/health")
async def health():
    return {"service": "nova", "status": "ok", "cosign_ready": bool(COSIGN_SECRET), "offline": OFFLINE, "version": NOVA_VERSION}

@app.post("/intent/lint")
async def lint(spec: Dict[str, Any] = Body(...)):
    intent = spec.get("intent") or ""
    payload = spec.get("payload") or {}
    reason = (payload.get("reason") or "").strip()
    issues = []
    if not reason:
        issues.append("missing_reason")
    elif not HUMAN_RE.search(reason):
        issues.append("non_human_reason")
    if intent == "pool.scale":
        reps = payload.get("replicas")
        if isinstance(reps, int) and reps > int(os.getenv("NOVA_MAX_REPLICAS_PER_CHANGE","5")):
            issues.append("replicas_too_high")
    if intent not in {"pool.scale","route.update","data.purge"}:
        issues.append("unknown_intent")
    return {"ok": len(issues)==0, "issues": issues, "canon_id": CANON_ID}

@app.post("/intent/cosign")
async def cosign(spec: Dict[str, Any] = Body(...)):
    if not COSIGN_SECRET:
        return {"ok": False, "error": "cosign_secret_missing"}
    endpoint = spec.get("endpoint","")
    payload = spec.get("payload", {})
    intent = spec.get("intent") or payload.get("intent") or ""
    lint_resp = await lint({"intent": intent, "payload": payload})
    if not lint_resp.get("ok"):
        return {"ok": False, "error": "lint_failed", "issues": lint_resp.get("issues", []), "canon_id": CANON_ID}
    raw = canonicalize(payload)
    token = hmac.new(COSIGN_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return {"ok": True, "endpoint": endpoint, "intent": intent, "cosign": token, "canon_id": CANON_ID}
