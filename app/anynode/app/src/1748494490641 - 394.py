
# C:\Engineers\root\scripts\bootstrap_environment.py
# Bootstrap script to detect, log, and respond to environment changes for Engineers

import os
import json
import platform
import psutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
CONTEXT_PATH = os.path.join(BASE_DIR, "..", "environment_context.json")

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
        with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
        print(f"Environment context saved to: {CONTEXT_PATH}")
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
