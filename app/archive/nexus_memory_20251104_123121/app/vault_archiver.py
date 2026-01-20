# 📂 Path: /Systems/engine/memory/vault_archiver.py

import os
import json
from datetime import datetime

def archive_log(log_path, vault_folder = '/memory/vault/backups/'):
    if not os.path.exists(log_path):
        print("[Vault Archiver] No log found to archive.")
        return

    with open(log_path, 'r') as f:
        try:
            data  =  json.load(f)
        except json.JSONDecodeError:
            print("[Vault Archiver] Corrupted log, cannot archive.")
            return

    if not data:
        print("[Vault Archiver] Log is empty, skipping archive.")
        return

    os.makedirs(vault_folder, exist_ok = True)
    archive_filename  =  f"vault_backup_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    archive_path  =  os.path.join(vault_folder, archive_filename)

    with open(archive_path, 'w') as archive_file:
        json.dump(data, archive_file, indent = 2)

    print(f"[Vault Archiver] Archived {len(data)} entries to {archive_path}")

    # Optionally clear original log after archive
    with open(log_path, 'w') as reset_file:
        reset_file.write('[]')

if __name__ == "__main__":
    archive_log('/memory/logs/guardian_memory_log.json')
