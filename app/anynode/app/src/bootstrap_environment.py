
# C:\Engineers\root\scripts\bootstrap_environment.py
# Bootstrap script to detect, log, and respond to environment changes for Engineers

import os
import json
import platform
import psutil
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.session_manager import load_sessions, save_session, append_to_latest_session
import common.session_weaviate as session_db

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
COMMON_DIR = os.path.join(ROOT_DIR, "common")
SESSION_MANAGER = os.path.join(COMMON_DIR, "session_manager.py")
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")
DB_PATH = os.path.join(MEMORY_DIR, "engineers.db")
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
MODEL_ASSIGNMENT = os.path.join(ROOT_DIR, "config", "model_assignment.json")
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

def collect_environment():
    env = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "model_directory_exists": os.path.exists(MODELS_DIR),
        "models_dir": MODELS_DIR,
        "model_files_count": 0,
        "gguf_files": [],
        "errors": []
    }

    if os.path.exists(MODELS_DIR):
        ggufs = []
        for root, _, files in os.walk(MODELS_DIR):
            for f in files:
                if f.endswith(".gguf"):
                    full_path = os.path.join(root, f)
                    ggufs.append(full_path)
        env["gguf_files"] = ggufs
        env["model_files_count"] = len(ggufs)
        if not ggufs:
            env["errors"].append("Model directory exists but contains no .gguf files.")
    else:
        env["errors"].append("Model directory does not exist.")

    return env

def save_context(context):
    try:
        with open(ENV_CONTEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
        print(f"Environment context saved to: {ENV_CONTEXT_FILE}")
    except Exception as e:
        print(f"Failed to write context file: {e}")

def main():
    print("Bootstrapping environment check...")
    env = collect_environment()
    save_context(env)
    if env["errors"]:
        print("Environment issues detected:")
        for err in env["errors"]:
            print("   -", err)
    else:
        print("Environment check passed. Ready for deployment.")

if __name__ == "__main__":
    main()
