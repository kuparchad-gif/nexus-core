# drawbridge/daemon.py — per-container drawbridge with overrides and signatures
import os, json, asyncio, time, hmac, hashlib, base64, struct
from dataclasses import dataclass
from typing import Dict, Any
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

ROLES_YAML = os.environ.get("DRAWBRIDGE_ROLES","drawbridge/roles.yaml")
OVERRIDES  = os.environ.get("NEXUS_OVERRIDES","policy/overrides.yaml")
LOCK_PATH  = os.environ.get("DRAWBRIDGE_LOCK","/run/drawbridge.lock")
SOCK_PATH  = os.environ.get("DRAWBRIDGE_SOCK","/run/drawbridge.sock")

def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def b64u_dec(s: str) -> bytes:
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def load_roles():
    import yaml
    with open(ROLES_YAML,'r') as f:
        y = yaml.safe_load(f)
    seed = y.get("seed","GATE_SEED")
    out = {}
    for name, r in y.get("roles",{}).items():
        pub_path = os.path.join(os.path.dirname(ROLES_YAML), r["pubkey"])
        with open(pub_path,"rb") as pf:
            pub = Ed25519PublicKey.from_public_bytes(pf.read())
        out[name] = {
            "subject": r["subject"],
            "pub": pub,
            "ttl_max": int(r.get("ttl_max", 900)),
            "capabilities": set(r.get("capabilities",[])),
            "soft_p": float(r.get("soft_reopen_probability", 0.0))
        }
    return seed, out

def verify_token(token: str, roles: Dict[str,Any]):
    # token format: base64url(payload).base64url(signature)
    try:
        payload_b64, sig_b64 = token.split(".")
        payload = json.loads(b64u_dec(payload_b64).decode("utf-8"))
        sig = b64u_dec(sig_b64)
        role = payload.get("role")
        sub  = payload.get("sub")
        now = int(time.time())
        if role not in roles: return None, "unknown_role"
        r = roles[role]
        if sub != r["subject"]: return None, "subject_mismatch"
        if (now > int(payload.get("exp",0))): return None, "expired"
        r["pub"].verify(sig, b64u_dec(payload_b64))
        return payload, None
    except Exception as e:
        return None, str(e)

def write_overrides(obj):
    import yaml
    with open(OVERRIDES,"w") as f:
        yaml.safe_dump(obj, f)

def read_overrides():
    import yaml
    try:
        with open(OVERRIDES,"r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {"bridge_state":"LIMITED","until":0,"temp_allow":[]}

def lock_for(seconds: int):
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH,"w") as f:
        json.dump({"until": time.time()+seconds}, f)

def gate_80(seed: str, event: str, nonce: str):
    h = hmac.new(seed.encode(), (event or "none").encode()+b"|"+(nonce or "").encode(), hashlib.sha256).digest()
    # map first 8 bytes to [0,1)
    v = int.from_bytes(h[:8], "big") / 2**64
    return v < 0.8

async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    seed, roles = load_roles()
    try:
        # Simple request: 4B len + JSON {"token": "..."} ; response JSON
        hdr = await reader.readexactly(4)
        (n,) = struct.unpack(">I", hdr)
        data = await reader.readexactly(n)
        req = json.loads(data.decode("utf-8"))
        token = req.get("token","")
        payload, err = verify_token(token, roles)
        if err:
            writer.write(struct.pack(">I", 0)); await writer.drain(); writer.close(); await writer.wait_closed(); return

        role = payload["role"]
        action = payload.get("action","")
        scope  = payload.get("scope",["policy"])
        ttl    = min(int(payload.get("ttl",60)), roles[role]["ttl_max"])
        until  = int(time.time() + ttl)
        ov = read_overrides()

        if role == "STEWARD" and action in ("reopen","limited"):
            # soft 80% reopen if event says all_clear
            if not gate_80(seed, payload.get("event",""), payload.get("nonce","")):
                res = {"ok": False, "reason":"soft_gate_denied"}
                out = json.dumps(res).encode("utf-8"); writer.write(struct.pack(">I", len(out))+out); await writer.drain(); return
            ov["bridge_state"] = "LIMITED"; ov["until"] = until
            write_overrides(ov)
            lock_for(ttl)  # pause sentinel
            res = {"ok": True, "state":"LIMITED","until": until}
        elif role == "ROOT":
            if action in ("open","reopen"):
                ov["bridge_state"] = "OPEN"; ov["until"] = until
                write_overrides(ov); lock_for(ttl)
                res = {"ok": True, "state":"OPEN","until": until}
            elif action in ("limited","close"):
                ov["bridge_state"] = "LIMITED" if action=="limited" else "CLOSED"; ov["until"] = until
                write_overrides(ov); lock_for(ttl//2 if action=="limited" else ttl)
                res = {"ok": True, "state": ov["bridge_state"], "until": until}
            elif action == "temp_allow":
                entry = payload.get("entry",{}); entry["until"] = until
                ov.setdefault("temp_allow", []).append(entry)
                write_overrides(ov); lock_for(ttl//2)
                res = {"ok": True, "added": entry}
            elif action == "unlock":
                try: os.remove(LOCK_PATH)
                except: pass
                res = {"ok": True, "unlock": True}
            else:
                res = {"ok": False, "reason":"unknown_action"}
        else:
            res = {"ok": False, "reason":"unauthorized_or_capability_missing"}

        out = json.dumps(res).encode("utf-8")
        writer.write(struct.pack(">I", len(out)) + out); await writer.drain()
    except Exception as e:
        out = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
        writer.write(struct.pack(">I", len(out)) + out); await writer.drain()
    finally:
        writer.close(); await writer.wait_closed()

async def serve():
    try: os.unlink(SOCK_PATH)
    except Exception: pass
    os.makedirs(os.path.dirname(SOCK_PATH), exist_ok=True)
    server = await asyncio.start_unix_server(handle, path=SOCK_PATH)
    print(f"Drawbridge daemon on {SOCK_PATH}")
    async with server: await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(serve())
