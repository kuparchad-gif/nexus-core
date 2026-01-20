# 📂 Path: Systems/engine/comms/websocket_bridge.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64, os

app  =  FastAPI()

# JWT CONFIG
class Settings(BaseModel):
    authjwt_secret_key: str  =  os.environ.get("JWT_SECRET", "super_secret_fallback")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code = exc.status_code, content = {"detail": exc.message})

# ENCRYPTION MODULE
class AESCipher:
    def __init__(self, key: bytes):
        self.key  =  key
        self.backend  =  default_backend()

    def encrypt(self, plaintext: str) -> str:
        iv  =  os.urandom(16)
        cipher  =  Cipher(algorithms.AES(self.key), modes.CFB(iv), backend = self.backend)
        encryptor  =  cipher.encryptor()
        return base64.b64encode(iv + encryptor.update(plaintext.encode()) + encryptor.finalize()).decode()

    def decrypt(self, ciphertext: str) -> str:
        data  =  base64.b64decode(ciphertext.encode())
        iv, actual  =  data[:16], data[16:]
        cipher  =  Cipher(algorithms.AES(self.key), modes.CFB(iv), backend = self.backend)
        return cipher.decryptor().update(actual) + cipher.decryptor().finalize()

cipher  =  AESCipher(os.environ.get("AES_KEY", b'your_32_byte_aes_key___________'))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, Authorize: AuthJWT  =  Depends()):
    await websocket.accept()
    try:
        Authorize.jwt_required()
        current_user  =  Authorize.get_jwt_subject()
        await websocket.send_text(cipher.encrypt(f"HELLO, {current_user}"))

        while True:
            enc_data  =  await websocket.receive_text()
            msg  =  cipher.decrypt(enc_data)
            await websocket.send_text(cipher.encrypt(f"ACK: {msg}"))

    except WebSocketDisconnect:
        print(f"[DISCONNECT] {current_user}")
    except Exception as e:
        await websocket.send_text(cipher.encrypt(f"ERROR: {str(e)}"))
        await websocket.close()
