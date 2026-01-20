# File: /root/boot/llm_watchdog.py

# Instructions
# - Monitors loaded LLMs via multiple backends (LM Studio, Ollama, vLLM)
# - Reloads models if they become unavailable
# - Configured via /config/environment_context.json
# - Logs to /logs/llm_watchdog.log

import os
import json
import time
import requests
from datetime import datetime

ASSIGNMENT_PATH = os.path.join("config", "model_assignment.json")
ENVIRONMENT_PATH = os.path.join("config", "environment_context.json")
LOG_FILE = os.path.join("logs", "llm_watchdog.log")
MODEL_DIR = os.path.join("models")
CHECK_INTERVAL = 60  # seconds

# Backend Endpoints
BACKENDS = {
    "lmstudio": "http://127.0.0.1:1313/v1/models",
    "ollama": "http://127.0.0.1:11434/api/tags",
    "vllm": "http://127.0.0.1:8000/v1/models"
}


def log(msg):
    ts = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[WATCHDOG] {msg}")


def get_environment():
    if os.path.exists(ENVIRONMENT_PATH):
        with open(ENVIRONMENT_PATH, 'r') as f:
            env = json.load(f)
            return env.get("llm_backend", "lmstudio")
    return "lmstudio"


def get_loaded_models(backend):
    try:
        url = BACKENDS.get(backend)
        if backend == "ollama":
            res = requests.get(url)
            if res.status_code == 200:
                return [m['name'] for m in res.json().get('models', [])]
        elif backend in ["lmstudio", "vllm"]:
            res = requests.get(url)
            if res.status_code == 200:
                return [m['id'] for m in res.json() if 'id' in m]
        log(f"ERROR: Failed to get models from {backend} ({res.status_code})")
    except Exception as e:
        log(f"ERROR: Exception while polling {backend}: {e}")
    return []


def reload_model(model_id, backend):
    try:
        if backend == "lmstudio":
            res = requests.post(f"{BACKENDS[backend]}/{model_id}/load")
        elif backend == "ollama":
            res = requests.post("http://127.0.0.1:11434/api/pull", json={"name": model_id})
        elif backend == "vllm":
            res = requests.post(f"{BACKENDS[backend]}/load", json={"model": model_id})
        if res.status_code == 200:
            log(f"✅ Reloaded model: {model_id}")
        else:
            log(f"❌ Reload failed for model: {model_id} ({res.status_code})")
    except Exception as e:
        log(f"🔥 Exception while reloading {model_id}: {e}")


def monitor_models():
    backend = get_environment()
    log(f"Using backend: {backend}")

    with open(ASSIGNMENT_PATH, "r", encoding="utf-8") as f:
        assignments = json.load(f)

    required_models = set()
    for models in assignments.values():
        required_models.update(models)

    while True:
        current = get_loaded_models(backend)
        for model_id in required_models:
            if model_id not in current:
                log(f"🔁 Model down: {model_id}. Reloading...")
                reload_model(model_id, backend)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    log("--- LLM Watchdog Started ---")
    monitor_models()
