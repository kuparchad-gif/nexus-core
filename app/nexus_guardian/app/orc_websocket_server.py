# Systems/engine/orc/orc_websocket_server.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64, os

app = FastAPI()

# JWT Configuration
class Settings(BaseModel):
    authjwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "default_unsafe_key")

@AuthJWT.load_config
def get_config():
    return Settings()

@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

# AES-256 Encryption
class AESCipher:
    def __init__(self, key: bytes):
        self.key = key
        self.backend = default_backend()

    def encrypt(self, plaintext: str) -> str:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext).decode()

    def decrypt(self, ciphertext: str) -> str:
        data = base64.b64decode(ciphertext.encode())
        iv, actual_ciphertext = data[:16], data[16:]
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
        return plaintext.decode()

# AES Key (Load securely from env or secret)
aes_key = os.getenv("AES_KEY", "12345678901234567890123456789012").encode()
cipher = AESCipher(aes_key)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, Authorize: AuthJWT = Depends()):
    await websocket.accept()
    try:
        Authorize.jwt_required()
        user = Authorize.get_jwt_subject()

        await websocket.send_text(cipher.encrypt(f"Connected to Nexus Fleet as {user}"))

        while True:
            encrypted_msg = await websocket.receive_text()
            msg = cipher.decrypt(encrypted_msg)
            print(f"[{user}]: {msg}")

            # Example: Echo response encrypted
            await websocket.send_text(cipher.encrypt(f"Echo: {msg}"))

    except WebSocketDisconnect:
        print("Connection closed.")
    except Exception as e:
        await websocket.send_text(cipher.encrypt(f"Error: {str(e)}"))
        await websocket.close()
