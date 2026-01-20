from cryptography.fernet import Fernet

class LocalDatabase:
    def __init__(self, security_layer):
        self.data = {}
        self.security_layer = security_layer

    def store(self, key: str, data: dict):
        encrypted_data = self.security_layer.encrypt_data(json.dumps(data))
        self.data[key] = encrypted_data

    def retrieve(self, key: str) -> dict:
        encrypted_data = self.data.get(key)
        if encrypted_data:
            return json.loads(self.security_layer.decrypt_data(encrypted_data))
        return None

    def delete(self, key: str):
        self.data.pop(key, None)