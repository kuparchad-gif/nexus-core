# File: /root/utils/fallback_model_loader.py

# Instructions
# Purpose: Provides logic to reload or reassign model endpoints when guardian reports failures
# Usage: Called by guardian or manually if a model is non-responsive
# Notes:
# - Requires `llm_guardian.py` to detect outages first
# - Self-healing hooks can call a local LLM loader script or external restore command

import os
import subprocess
from common.session_manager import append_to_latest_session

FALLBACK_COMMANDS = {
    "memory": ["python", "Systems/memory/scripts/reload_model.py"],
    "planner": ["python", "Systems/planner/scripts/reload_model.py"],
    "orc": ["python", "Systems/orc/scripts/reload_model.py"],
    "myth": ["python", "Systems/myth/scripts/reload_model.py"],
    "nexus": ["python", "Systems/nexus/scripts/reload_model.py"]
    # Extend as needed
}

def try_fallback(llm_name):
    command = FALLBACK_COMMANDS.get(llm_name)
    if not command:
        print(f"[SKIP] No fallback command for: {llm_name}")
        return False
    try:
        subprocess.run(command, check=True)
        append_to_latest_session("llm_recovery", {
            "llm": llm_name,
            "recovered": True
        })
        print(f"[RECOVERY] {llm_name} reloaded.")
        return True
    except Exception as e:
        append_to_latest_session("llm_recovery", {
            "llm": llm_name,
            "recovered": False,
            "error": str(e)
        })
        print(f"[FAILED] {llm_name} recovery error:", e)
        return False

if __name__ == "__main__":
    for llm in FALLBACK_COMMANDS:
        try_fallback(llm)
