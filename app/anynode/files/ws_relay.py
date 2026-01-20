# /Systems/engine/orc/ws_relay.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64, os

app  =  FastAPI()

# ──────────────────────────────────────
# JWT Settings
# ──────────────────────────────────────
class Settings(BaseModel):
    authjwt_secret_key: str  =  os.getenv("JWT_SECRET", "super-secret-default")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def jwt_error_handler(request, exc):
    return JSONResponse(status_code = exc.status_code, content = {"detail": exc.message})

# ──────────────────────────────────────
# AES-256 Cipher for payload encryption
# ──────────────────────────────────────
class AESCipher:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("AES key must be 32 bytes long for AES-256.")
        self.key  =  key
        self.backend  =  default_backend()

    def encrypt(self, plaintext: str) -> str:
        iv  =  os.urandom(16)
        cipher  =  Cipher(algorithms.AES(self.key), modes.CFB(iv), backend = self.backend)
        encryptor  =  cipher.encryptor()
        ciphertext  =  encryptor.update(plaintext.encode()) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext).decode()

    def decrypt(self, ciphertext: str) -> str:
        data  =  base64.b64decode(ciphertext.encode())
        iv  =  data[:16]
        actual_ciphertext  =  data[16:]
        cipher  =  Cipher(algorithms.AES(self.key), modes.CFB(iv), backend = self.backend)
        decryptor  =  cipher.decryptor()
        plaintext  =  decryptor.update(actual_ciphertext) + decryptor.finalize()
        return plaintext.decode()

# AES Key pulled securely
aes_key  =  os.getenv("AES_KEY", "Th!s1sMyUltraSecureAesKey32Byte!")[:32].encode()
cipher  =  AESCipher(aes_key)

# ──────────────────────────────────────
# WebSocket Secure Handler
# ──────────────────────────────────────
@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket, Authorize: AuthJWT  =  Depends()):
    await websocket.accept()
    try:
        Authorize.jwt_required()
        current_user  =  Authorize.get_jwt_subject()
        await websocket.send_text(cipher.encrypt(f"👋 Welcome {current_user}! Encrypted link secured."))

        while True:
            encrypted_message  =  await websocket.receive_text()
            decrypted  =  cipher.decrypt(encrypted_message)
            # Process your command, route, or signal
            response  =  f"🛰️ Relayed: {decrypted}"
            await websocket.send_text(cipher.encrypt(response))

    except WebSocketDisconnect:
        print(f"[⚡] WebSocket disconnected: {current_user}")
    except Exception as e:
        await websocket.send_text(cipher.encrypt(f"[ERROR] {str(e)}"))
        await websocket.close()
