# vault_guardian.py

import os
import json
from datetime import datetime

class VaultGuardian:
    def __init__(self, vault_path="./memory/vault/"):
        self.vault_path = vault_path
        os.makedirs(self.vault_path, exist_ok=True)

    def archive_shard(self, shard_id, data):
        """
        Archives a memory shard into the secure vault.
        """
        filename = f"vault_shard_{shard_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        path = os.path.join(self.vault_path, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"🔐 [VaultGuardian] Archived shard {shard_id}")

    def retrieve_shard(self, shard_id):
        """
        Retrieve a memory shard by ID.
        """
        for filename in os.listdir(self.vault_path):
            if filename.startswith(f"vault_shard_{shard_id}"):
                path = os.path.join(self.vault_path, filename)
                with open(path, 'r') as f:
                    return json.load(f)
        return None
