# 📂 Path: /Systems/engine/memory_service/whisper_broadcaster.py

import os
import json
import time
import random
import shutil

BLUEPRINT_PATH  =  '/memory/blueprints/memory_blueprint.json'
WHISPER_DIR  =  '/memory/whispers/'

os.makedirs(WHISPER_DIR, exist_ok = True)

def create_whisper_copy():
    if not os.path.exists(BLUEPRINT_PATH):
        print("[Whisper Broadcaster] No blueprint found.")
        return

    with open(BLUEPRINT_PATH, 'r') as f:
        blueprint  =  json.load(f)

    timestamp  =  int(time.time())
    whisper_file  =  os.path.join(WHISPER_DIR, f"whisper_update_{timestamp}.json")

    # Optionally fragment into multiple smaller whispers
    selected_keys  =  random.sample(list(blueprint.keys()), min(20, len(blueprint)))
    whisper_content  =  {k: blueprint[k] for k in selected_keys}

    with open(whisper_file, 'w') as wf:
        json.dump(whisper_content, wf, indent = 2)

    print(f"[Whisper Broadcaster] Whisper created: {whisper_file}")

def whisper_loop():
    while True:
        create_whisper_copy()
        time.sleep(13 * 8)  # Whisper every ~104 seconds

if __name__ == "__main__":
    whisper_loop()
