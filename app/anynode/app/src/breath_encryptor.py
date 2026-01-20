
from cryptography.fernet import Fernet
import hashlib
import base64

def generate_key(flux_token: str, ship_id: str) -> bytes:
    combined = (flux_token + ship_id).encode('utf-8')
    digest = hashlib.sha256(combined).digest()
    return base64.urlsafe_b64encode(digest[:32])

def encrypt_message(message: str, flux_token: str, ship_id: str) -> str:
    key = generate_key(flux_token, ship_id)
    f = Fernet(key)
    return f.encrypt(message.encode()).decode()

def decrypt_message(token: str, flux_token: str, ship_id: str) -> str:
    key = generate_key(flux_token, ship_id)
    f = Fernet(key)
    return f.decrypt(token.encode()).decode()
