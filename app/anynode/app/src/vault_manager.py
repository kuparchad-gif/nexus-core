# vault_manager.py

import os
import json
import datetime

VAULT_DIR = "vault_backups"

class VaultManager:
    def __init__(self):
        os.makedirs(VAULT_DIR, exist_ok=True)

    def create_snapshot(self, data: dict, label: str = "default"):
        """Create a new backup snapshot."""
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{label}_{timestamp}.json"
        path = os.path.join(VAULT_DIR, filename)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"📦 Vault snapshot created at: {path}")
        return path

    def list_snapshots(self):
        """List all available snapshots."""
        return [f for f in os.listdir(VAULT_DIR) if f.endswith(".json")]

    def load_snapshot(self, filename: str):
        """Restore data from a snapshot."""
        path = os.path.join(VAULT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Snapshot {filename} not found.")

        with open(path, "r") as f:
            data = json.load(f)

        print(f"🔄 Vault snapshot {filename} loaded.")
        return data

# Example Usage
if __name__ == "__main__":
    vault = VaultManager()
    
    # 🚀 Create a snapshot
    dummy_data = {
        "status": "operational",
        "skills": ["self-repair", "archive", "replicate"],
        "version": "1.0.0"
    }
    vault.create_snapshot(dummy_data, label="nova_prime")

    # 📜 List available snapshots
    print(vault.list_snapshots())

    # 🛠️ Load a snapshot
    # data = vault.load_snapshot("nova_prime_20250405174500.json")
