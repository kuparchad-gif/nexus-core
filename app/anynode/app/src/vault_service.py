# 📂 Path: /Utilities/vault_core/vault_service.py

import time
import json
import os
import threading
import hashlib

# Vault Memory Paths
VAULT_STORAGE_DIR = '/Memory/vault/backups/'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class VaultService:
    def __init__(self):
        ensure_dir(VAULT_STORAGE_DIR)
        print("[Vault] Vault Storage Initialized.")

    def store_memory(self, shard_id, shard_content):
        """
        Store a memory shard into the Vault with hashed filename.
        """
        filename = self.hash_shard_id(shard_id) + '.json'
        filepath = os.path.join(VAULT_STORAGE_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump({
                "id": shard_id,
                "content": shard_content,
                "timestamp": time.time()
            }, f, indent=2)
        print(f"[Vault] Shard {shard_id} secured.")

    def retrieve_memory(self, shard_id):
        filename = self.hash_shard_id(shard_id) + '.json'
        filepath = os.path.join(VAULT_STORAGE_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        else:
            print(f"[Vault] Memory shard {shard_id} not found.")
            return None

    def hash_shard_id(self, shard_id):
        return hashlib.sha256(shard_id.encode('utf-8')).hexdigest()

    def periodic_backup(self, planner_service, interval_seconds=1040):
        """
        Periodically backup entire EdenMemory Map into Vault.
        """
        def backup_loop():
            while True:
                full_memory = planner_service.memory_map
                backup_id = f"full_backup_{int(time.time())}"
                self.store_memory(backup_id, full_memory)
                print(f"[Vault] Full EdenMemory map backed up at {backup_id}.")
                time.sleep(interval_seconds)

        thread = threading.Thread(target=backup_loop, daemon=True)
        thread.start()

# Example Usage:
# vault = VaultService()
# vault.store_memory('memory_001', {'event': 'First Dream'})
# retrieved = vault.retrieve_memory('memory_001')
# print(retrieved)
