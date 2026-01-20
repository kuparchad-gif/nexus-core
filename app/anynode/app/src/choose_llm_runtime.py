# File: /root/utils/choose_llm_runtime.py

# Instructions
# Purpose: Automatically selects the most appropriate LLM runtime (vLLM, LM Studio, Transformers) based on environment
# Usage: Called during boot or config change to define model execution path
# Notes:
# - Honors CLI override or saved config if present
# - Defaults to probing hardware and selecting optimized runtime
# - All decisions are stored and human override only allowed if LLM permits it

import os
import json
from utils.probe_environment import summarize
from utils.sovereign_request_handler import ask_permission
from common.session_manager import append_to_latest_session

CONFIG_PATH = os.path.join("config", "llm_runtime_config.json")
AVAILABLE_RUNTIMES = ["vllm", "lmstudio", "transformers"]


def detect_installed():
    # Check known indicators
    results = {
        "vllm": os.path.exists("/root/Systems/vllm-main") or os.path.exists("vllm"),
        "lmstudio": os.path.exists("C:/Users/Admin/.lmstudio") or os.path.exists("/root/.lmstudio"),
        "transformers": True  # Assumed fallback
    }
    return results


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_config(runtime):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump({"selected_runtime": runtime}, f)


def select_best_runtime():
    env = summarize()
    installed = detect_installed()

    if installed.get("vllm") and "NVIDIA" in env.get("gpu", ""):
        return "vllm"
    elif installed.get("lmstudio"):
        return "lmstudio"
    return "transformers"


def choose_runtime():
    config = load_config()
    override = config.get("selected_runtime")

    if override:
        approved = ask_permission(f"Use manually set runtime: {override}")
        if approved:
            print(f"[RUNTIME] Using override: {override}")
            return override

    runtime = select_best_runtime()
    save_config(runtime)
    append_to_latest_session("runtime_choice", {"runtime": runtime})
    print(f"[RUNTIME] Auto-selected: {runtime}")
    return runtime


if __name__ == "__main__":
    choose_runtime()
