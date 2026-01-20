# drawbridge/keygen.py — create ed25519 keypair
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import sys
name = sys.argv[1] if len(sys.argv)>1 else "user"
sk = Ed25519PrivateKey.generate()
pk = sk.public_key()
open(f"drawbridge/keys/{name}.priv","wb").write(
    sk.private_bytes(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
)
open(f"drawbridge/keys/{name}.pub","wb").write(
    pk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
)
print("Wrote drawbridge/keys/%s.priv and .pub" % name)
