# 📂 Path: /Systems/engine/orc/orc_memory_uploader.py

import requests

MEMORY_ENDPOINT = "http://localhost:8081/upload_memory/"  # Adjust to Memory Service IP

def push_text_memory(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        response = requests.post(MEMORY_ENDPOINT, files=files)

    if response.ok:
        print("[Orchestrator Uploader] Memory pushed successfully.")
    else:
        print("[Orchestrator Uploader] Upload failed:", response.text)
