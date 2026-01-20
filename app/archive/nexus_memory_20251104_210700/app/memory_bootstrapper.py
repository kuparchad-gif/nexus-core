# 📂 Path: /Systems/engine/memory/memory_bootstrapper.py

import os

# Define necessary memory paths
memory_paths  =  [
    '/memory/bootstrap/genesis',
    '/memory/logs',
    '/memory/vault/backups',
    '/memory/memory/indexer',
    '/memory/memory/lineage',
    '/memory/memory/streams',
    '/memory/subconscious'
]

def ensure_memory_structure():
    for path in memory_paths:
        try:
            os.makedirs(path, exist_ok = True)
            print(f"[Memory Bootstrap] Verified: {path}")
        except Exception as e:
            print(f"[Memory Bootstrap] ERROR creating {path}: {str(e)}")

if __name__ == "__main__":
    ensure_memory_structure()
