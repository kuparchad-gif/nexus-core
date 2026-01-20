# 📂 Path: /Systems/engine/memory_service/pulse_sync_engine.py

import time
import shutil

def sync_blueprint():
    src  =  '/memory/blueprints/memory_blueprint.json'
    dst  =  '/memory/vault/backups/memory_blueprint_sync.json'

    if os.path.exists(src):
        shutil.copy(src, dst)
        print("[Pulse Sync Engine] Blueprint synced to vault.")

def heartbeat_loop():
    pulse  =  0
    while True:
        pulse + =  1
        if pulse % 104 == 0:
            sync_blueprint()
        time.sleep(13)

if __name__ == "__main__":
    heartbeat_loop()
