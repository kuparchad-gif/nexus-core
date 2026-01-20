from __future__ import annotations
import os, base64, json, hashlib
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

class WalletVault:
    def __init__(self, path: str, master_password: str):
        self.path = path
        self.master_password = master_password.encode()
        self._fernet = None

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password))
        return key

    def _fernet_from_password(self, salt: bytes) -> Fernet:
        return Fernet(self._derive_key(salt))

    def unlock(self):
        if not os.path.exists(self.path):
            # initialize new vault
            salt = os.urandom(16)
            with open(self.path, "wb") as f:
                f.write(salt + b"::" + b"")
        with open(self.path, "rb") as f:
            raw = f.read()
        salt, payload = raw.split(b"::", 1)
        self._fernet = self._fernet_from_password(salt)
        return True

    def store_seed(self, mnemonic: str):
        assert self._fernet is not None, "vault locked"
        salt = open(self.path, "rb").read().split(b"::",1)[0]
        token = self._fernet.encrypt(mnemonic.encode())
        with open(self.path, "wb") as f:
            f.write(salt + b"::" + token)

    def load_seed(self) -> str:
        assert self._fernet is not None, "vault locked"
        salt, token = open(self.path, "rb").read().split(b"::",1)
        return self._fernet.decrypt(token).decode()
