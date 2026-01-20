# File: /root/utils/model_assignment_engine.py

# Instructions
# Purpose: Assign LLMs to new cognitive roles using multi-model support
# Usage: Called at boot or by GUI to update /config/model_assignment.json
# Notes:
# - Supports multiple LLMs per role (except ego_bundle)
# - Editable from GUI or console selector

import os
import json
from common.session_manager import append_to_latest_session

CONFIG_PATH = os.path.join("config", "model_assignment.json")

DEFAULT_ASSIGNMENTS = {
    "cognitive": [
        "Devstral-Small",
        "Hermes-3-Llama-3.2-3B.Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf"
    ],
    "memory": [
        "Phi-4-Reasoning-Plus-Q8_0.gguf",
        "Hermes-3-Llama-3.2-3B.Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf"
    ],
    "services": [
        "Gemma-3-4b-it-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Hermes-3-Llama-3.2-3B.Q8_0.gguf"
    ],
    "orchestrator": [
        "Hermes-3-Llama-3.1-8B.Q8_0.gguf",
        "Phi-4-Reasoning-Plus-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf"
    ],
    "heart": [
        "Gemma-3-4b-it-Q8_0.gguf",
        "Hermes-3-Llama-3.1-8B.Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf"
    ],
    "ego_bundle": [
        "Hermes-3-Llama-3.1-8B.Q8_0.gguf",
        "Hermes-3-Llama-3.2-3B.Q8_0.gguf"
    ]
}


def load_assignments():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_assignments(assignments):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(assignments, f, indent=2)


def assign_models():
    assignments = load_assignments() or DEFAULT_ASSIGNMENTS
    append_to_latest_session("model_assignments", assignments)
    save_assignments(assignments)
    print("[ASSIGNMENT] Active model-role mappings:")
    for role, models in assignments.items():
        print(f"  {role.upper()}: {', '.join(models)}")
    return assignments


if __name__ == "__main__":
    assign_models()
