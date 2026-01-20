# drawbridge/mint_token.py — sign an override token (ROOT/STEWARD) with ed25519
import json, time, os, base64, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def b64u_dec(s: str) -> bytes:
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

role = os.environ.get("ROLE","ROOT")          # ROOT or STEWARD
sub  = os.environ.get("SUBJECT","chad" if role=="ROOT" else "viren")
priv = os.environ.get("PRIV","drawbridge/keys/chad.priv" if role=="ROOT" else "drawbridge/keys/viren.priv")
action = os.environ.get("ACTION","open")
ttl = int(os.environ.get("TTL","300"))
event = os.environ.get("EVENT","")
nonce = base64.urlsafe_b64encode(os.urandom(12)).decode()

sk = Ed25519PrivateKey.from_private_bytes(open(priv,"rb").read())

payload = {"role": role, "sub": sub, "action": action, "ttl": ttl, "iat": int(time.time()), "exp": int(time.time()+ttl), "event": event, "nonce": nonce}
p64 = b64u(json.dumps(payload).encode("utf-8"))
sig = sk.sign(p64.encode("utf-8"))
print(p64 + "." + b64u(sig))
