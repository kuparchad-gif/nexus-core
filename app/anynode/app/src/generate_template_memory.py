
# C:\Engineers\eden_engineering\scripts\memory_initializer.py
# Generates a clean, pre-identity system memory snapshot for Engineer-based seed AIs
# Use this to capture a snapshot of the system before identity is injected.

import os
import json
import platform
import psutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "..", "public")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
CONTEXT_PATH = os.path.join(BASE_DIR, "..", "environment_context.json")
LOADED_MODELS_JSON = os.path.join(PUBLIC_DIR, "loaded_models.json")
TEMPLATE_MEMORY_PATH = os.path.join(BASE_DIR, "..", "template_engineer_memory.json")

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def gather_memory():
    memory = {
        "template_generated": datetime.now().isoformat(),
        "system": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "cpu": platform.processor(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version(),
        },
        "models_loaded": load_json(LOADED_MODELS_JSON),
        "environment_context": load_json(CONTEXT_PATH),
        "files_present": [],
        "scripts_present": [],
        "roles_active": [],
        "status": "template",
        "identity_seed": None
    }

    # Scan core scripts
    scripts_path = os.path.join(BASE_DIR)
    if os.path.exists(scripts_path):
        memory["scripts_present"] = [
            f for f in os.listdir(scripts_path)
            if os.path.isfile(os.path.join(scripts_path, f)) and f.endswith(".py")
        ]

    # Check models available
    if os.path.exists(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for f in files:
                if f.endswith(".gguf"):
                    memory["files_present"].append(os.path.join(root, f))

    return memory

def save_memory(memory):
    try:
        with open(TEMPLATE_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        print(f"✅ Template memory saved to: {TEMPLATE_MEMORY_PATH}")
    except Exception as e:
        print("❌ Failed to write memory file:", e)

def main():
    print("🧠 Generating base Engineer memory...")
    mem = gather_memory()
    save_memory(mem)
    print("✅ Memory seed ready for identity injection.")

if __name__ == "__main__":
    main()
