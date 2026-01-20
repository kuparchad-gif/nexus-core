# C:\Engineers\root\scripts\memory_initializer.py
# Initializes the Engineers' memory at system startup

import os
import json
from datetime import datetime

# Define key file paths (project-root relative)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCROLL_PATH = os.path.join(ROOT, "public", "data", "viren_scroll.md")
TECH_MANIFEST_PATH = os.path.join(ROOT, "public", "data", "viren_technical_manifest.md")
ENV_CONTEXT_PATH = os.path.join(ROOT, "scripts", "environment_context.json")
LOG_DIR = os.path.join(ROOT, "logs", "boot_history")
OUTPUT_PATH = os.path.join(ROOT, "memory", "runtime", "engineers_state.json")

# Add step functions for program_launcher.py
def step_1_environment_setup():
    print("[STEP 1] Setting up environment...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    return {"root_path": ROOT, "status": "initialized"}

def step_2_memory_initialization():
    print("[STEP 2] Initializing memory...")
    main()
    return {"memory_path": OUTPUT_PATH, "status": "initialized"}

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def read_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def collect_log_snippets(log_dir, max_chars=5000):
    if not os.path.isdir(log_dir):
        return ""
    logs = []
    for fname in sorted(os.listdir(log_dir)):
        full = os.path.join(log_dir, fname)
        if os.path.isfile(full):
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                snippet = f.read()[-max_chars:]  # last N chars of each file
                logs.append(f"--- {fname} ---\n{snippet}")
    return "\n\n".join(logs)

def main():
    scroll = read_file(SCROLL_PATH)
    tech = read_file(TECH_MANIFEST_PATH)
    context = read_json(ENV_CONTEXT_PATH)
    logs = collect_log_snippets(LOG_DIR)

    state = {
        "timestamp": datetime.now(datetime.UTC).isoformat(),
        "scroll": scroll,
        "technical_manifest": tech,
        "environment_context": context,
        "boot_logs": logs,
        "source": "memory_initializer.py",
        "status": "bootstrapped"
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"[✅] Engineers memory state saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
