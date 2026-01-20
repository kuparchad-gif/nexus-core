# crypto_box.py
import os, base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

def load_key_from_env(var: str) -> bytes | None:
    val = os.getenv(var, "").strip()
    if not val:
        return None
    return base64.b64decode(val)

def seal(plaintext: bytes, key: bytes | None, cipher: str = "chacha20") -> bytes:
    if not key or cipher == "none":
        return plaintext
    if cipher == "aesgcm":
        aes = AESGCM(key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, None)
        return b"AGCM" + nonce + ct
    # default chacha20-poly1305
    ch = ChaCha20Poly1305(key)
    nonce = os.urandom(12)
    ct = ch.encrypt(nonce, plaintext, None)
    return b"CH20" + nonce + ct

def open(ct: bytes, key: bytes | None, cipher_hint: str = "chacha20") -> bytes:
    if not key or cipher_hint == "none":
        return ct
    tag = ct[:4]
    nonce = ct[4:16]
    body = ct[16:]
    if tag == b"AGCM":
        aes = AESGCM(key)
        return aes.decrypt(nonce, body, None)
    elif tag == b"CH20":
        ch = ChaCha20Poly1305(key)
        return ch.decrypt(nonce, body, None)
    else:
        # unknown tag, try hinted cipher
        if cipher_hint == "aesgcm":
            aes = AESGCM(key); return aes.decrypt(nonce, body, None)
        ch = ChaCha20Poly1305(key); return ch.decrypt(nonce, body, None)
