# AEAD wrapper for Nexus/NIM (ChaCha20-Poly1305) + HKDF key schedule
# pip install cryptography>=42
import os
from dataclasses import dataclass
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

AAD_DEFAULT = b"NEXUS:NIM:v1"

def hkdf_expand(master_key: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    hk = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    return hk.derive(master_key)

@dataclass
class Session:
    # 12-byte nonce = 4-byte sess_id + 8-byte counter
    key: bytes          # 32B subkey for AEAD
    sess_id: bytes      # 4B
    counter: int = 0
    role: bytes = b"S"  # b'S' server, b'C' client

    def next_nonce(self) -> bytes:
        self.counter = (self.counter + 1) & 0xFFFFFFFFFFFFFFFF
        return self.sess_id + self.counter.to_bytes(8, "big")

def new_session(master_key: bytes, context: bytes, role: bytes) -> "Session":
    if len(master_key) != 32:
        raise ValueError("master_key must be 32 bytes")
    salt = os.urandom(16)
    subkey = hkdf_expand(master_key, salt, context + role, 32)
    sess_id = hkdf_expand(master_key, salt, b"id"+context+role, 4)
    return Session(subkey, sess_id, 0, role)

def seal(session: "Session", plaintext: bytes, aad: bytes = AAD_DEFAULT) -> bytes:
    aead = ChaCha20Poly1305(session.key)
    nonce = session.next_nonce()  # unique per message per role
    ct = aead.encrypt(nonce, plaintext, aad + session.role)
    return session.sess_id + nonce[4:] + ct  # [sess_id||counter||ciphertext]

def open(session: "Session", sealed: bytes, aad: bytes = AAD_DEFAULT) -> bytes:
    if len(sealed) < 12 + 16:
        raise ValueError("sealed too short")
    sess_id = sealed[:4]
    if sess_id != session.sess_id:
        raise ValueError("session id mismatch")
    nonce = sess_id + sealed[4:12]
    ct = sealed[12:]
    aead = ChaCha20Poly1305(session.key)
    return aead.decrypt(nonce, ct, aad + session.role)
