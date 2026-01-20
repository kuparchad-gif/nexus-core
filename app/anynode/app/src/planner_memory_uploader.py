# 📂 Path: /Systems/engine/planner/planner_memory_uploader.py

import requests

MEMORY_ENDPOINT = "https://nexus-memory-687883244606.us-central1.run.app"  # Adjust to Memory Service IP

def push_text_memory(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        response = requests.post(MEMORY_ENDPOINT, files=files)

    if response.ok:
        print("[Planner Uploader] Memory pushed successfully.")
    else:
        print("[Planner Uploader] Upload failed:", response.text)
