from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from importlib import import_module
import os, time, hmac, hashlib, socket, json

try:
    import httpx
except Exception:
    httpx = None

TITLE = "Aethereal Nexus — Microcell"
VERSION = "0.1.0"

def _import_obj(spec: str):
    if ":" in spec:
        mod, attr = spec.split(":", 1)
    else:
        mod, attr = spec, "app"
    m = import_module(mod)
    return getattr(m, attr)

def _hmac(secret: str, msg: str) -> str:
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

def build_app() -> FastAPI:
    name = os.getenv("MICROCELL_NAME", "cell-unnamed")
    spec = os.getenv("MICROCELL_APP", "src.lilith.spine.relay_pi:app")
    caps = os.getenv("MICROCELL_CAPS", "relay").split(",")
    port = int(os.getenv("PORT", "9000"))
    secret = os.getenv("NEXUS_JWT_SECRET", "change-me")
    registry = os.getenv("REGISTRY_URL")  # e.g., http://registry:8500
    public_url = os.getenv("PUBLIC_URL")  # optional external URL

    root = FastAPI(title=f"{TITLE} ({name})", version=VERSION)
    # mount target app at root
    try:
        target = _import_obj(spec)
        root.mount("/", target)
        ok = True; err = None
    except Exception as e:
        ok = False; err = str(e)

    @root.get("/__cell__/health")
    def health():
        return {
            "status": "ok" if ok else "error",
            "name": name,
            "app_spec": spec,
            "caps": caps,
            "port": port,
            "mounted": ok,
            "error": err,
            "host": socket.gethostname(),
            "ts": time.time(),
        }

    @root.get("/__cell__/whoami")
    def whoami():
        return {"name": name, "caps": caps, "app_spec": spec, "public_url": public_url}

    # Best-effort registry announce (no crash if missing)
    if registry and httpx is not None:
        try:
            base = public_url or f"http://{os.getenv('HOSTNAME', 'localhost')}:{port}"
            token = _hmac(secret, f"{name}:{port}")
            payload = {"name": name, "caps": caps, "url": base, "token": token, "ts": time.time()}
            # fire-and-forget
            try:
                httpx.post(f"{registry.rstrip('/')}/register", json=payload, timeout=2.0)
            except Exception:
                pass
        except Exception:
            pass

    return root

app = build_app()
