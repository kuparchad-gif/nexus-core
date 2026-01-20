# secretkeeper/vault_init.py

import os
import json
from cryptography.fernet import Fernet

SECRETS_FILE = "secretkeeper/ship_secrets.json"

class VaultInit:
    def __init__(self, master_key=None):
        if master_key is None:
            # Try to load from ENV if not given
            master_key = os.getenv("NOVA_MASTER_KEY")
        if not master_key:
            raise ValueError("Master encryption key is missing.")

        self.fernet = Fernet(master_key.encode())
        os.makedirs(os.path.dirname(SECRETS_FILE), exist_ok=True)
        if not os.path.exists(SECRETS_FILE):
            with open(SECRETS_FILE, "w") as f:
                json.dump({}, f)

    def _load_secrets(self):
        with open(SECRETS_FILE, "r") as f:
            return json.load(f)

    def _save_secrets(self, secrets):
        with open(SECRETS_FILE, "w") as f:
            json.dump(secrets, f, indent=2)

    def create_secret(self, name, value):
        secrets = self._load_secrets()
        secrets[name] = self.fernet.encrypt(value.encode()).decode()
        self._save_secrets(secrets)
        print(f"🔐 Secret '{name}' created.")

    def retrieve_secret(self, name):
        secrets = self._load_secrets()
        encrypted = secrets.get(name)
        if not encrypted:
            print(f"⚠️ Secret '{name}' not found.")
            return None
        return self.fernet.decrypt(encrypted.encode()).decode()

    def update_secret(self, name, new_value):
        self.create_secret(name, new_value)
        print(f"🔄 Secret '{name}' updated.")

    def delete_secret(self, name):
        secrets = self._load_secrets()
        if name in secrets:
            del secrets[name]
            self._save_secrets(secrets)
            print(f"🗑️ Secret '{name}' deleted.")

# Example usage (only when run standalone)
if __name__ == "__main__":
    master_key = Fernet.generate_key().decode()
    os.environ["NOVA_MASTER_KEY"] = master_key
    print(f"🔑 Generated Master Key: {master_key}")

    vault = VaultInit(master_key)

    vault.create_secret("together_ai_key", "supersecret_value")
    print(vault.retrieve_secret("together_ai_key"))
