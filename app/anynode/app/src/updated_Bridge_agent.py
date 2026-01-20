# File: C:\Engineers\root\scripts\lmstudio_bridge.py
import os
import requests
import sys
from ..scripts.session_manager import load_sessions, save_session, append_to_latest_session

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")
DB_PATH = os.path.join(MEMORY_DIR, "engineers.db")
LOADED_MODELS_JSON = os.path.join(PUBLIC_DIR, "loaded_models.json")
LOAD_LIST = os.path.join(BASE_DIR, 'models_to_load.txt')
MANIFEST_PATH = r"C:\\Engineers\\root\\models\\model_manifest.json"
LOG_STREAM_FILE = r"C:\\Engineers\\root\\logs\\lms_prompt_debug.log"
ENV_CONTEXT_FILE = os.path.join(BASE_DIR, 'environment_context.json')
MAX_LOG_SIZE_MB = 20
LOAD_WAIT_SECS = 6
LMS_EXECUTABLE = "lms"
LMS_CWD = r"C:\\Engineers\\root\\Local\\Programs\\LM Studio\\resources\\app\\.webpack"
ANSI_SUPPORTED = sys.stdout.isatty() and os.name != "nt"
LMSTUDIO_BASE_URL = "http://localhost:1313"

def get_available_models():
    url = f"{LMSTUDIO_BASE_URL}/v1/models"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to retrieve models: {e}")
        return []

def query_model(model_id: str, prompt: str) -> str:
    # Decide endpoint based on model name pattern
    if any(tag in model_id.lower() for tag in ["chat", "llama", "mistral", "tulu", "vicuna", "gpt", "zephyr"]):
        endpoint = "chat/completions"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7
        }
    else:
        endpoint = "completions"
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7
        }

    url = f"{LMSTUDIO_BASE_URL}/v1/{endpoint}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if "chat" in endpoint:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            return result.get("choices", [{}])[0].get("text", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"❌ Query failed for model {model_id}: {e}")
        return f"Error: {model_id} failed to respond."
