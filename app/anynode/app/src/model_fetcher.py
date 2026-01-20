# model_fetcher.py
# Purpose: Utility for discovering running local models via LM Studio or Ollama
# Location: /root/bridge/model_fetcher.py

import os
import json
import requests
import logging

MANIFEST_PATH = r"C:\\Engineers\\root\\models\\model_manifest.json"
OUTPUT_PATH = r"C:\\Engineers\\root\\models\\loaded_models.json"
LMSTUDIO_BASE_URL = "http://localhost:1313"
OLLAMA_URL = "http://localhost:11434/api/tags"

logger = logging.getLogger("model_fetcher")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/boot_logs/model_fetcher.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def read_manifest():
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read manifest: {e}")
        return []

def get_lmstudio_models():
    try:
        response = requests.get(f"{LMSTUDIO_BASE_URL}/v1/models")
        if response.status_code == 200:
            return [m.get("id", "unknown") for m in response.json().get("data", [])]
    except Exception as e:
        logger.warning(f"LM Studio not reachable: {e}")
    return []

def get_ollama_models():
    try:
        response = requests.get(OLLAMA_URL)
        if response.status_code == 200:
            return [m.get("name", "unknown") for m in response.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Ollama not reachable: {e}")
    return []

def fetch_running_models():
    manifest = read_manifest()
    lmstudio_active = set(get_lmstudio_models())
    ollama_active = set(get_ollama_models())

    running = []
    for model in manifest:
        name = model.get("id") or model.get("name")
        source = model.get("source", "").lower()
        backend = "unknown"

        if name in lmstudio_active:
            backend = "lmstudio"
        elif name in ollama_active:
            backend = "ollama"

        if backend != "unknown":
            running.append({
                "id": name,
                "backend": backend,
                "size": model.get("size"),
                "architecture": model.get("architecture")
            })

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(running, f, indent=2)
        logger.info(f"Saved {len(running)} models to {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to write loaded models: {e}")

    return running

if __name__ == "__main__":
    for model in fetch_running_models():
        print(f"{model['id']} [{model['backend']}] — {model['architecture']} — {model['size']}")
