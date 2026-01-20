# llm_control.py
# Purpose: Routes model selection, handles LLM toggling, governs fallback, tracks environment state, and logs to file.
# Location: /Systems/engine/core/router/llm_control.py
# Requires: /config/model_compatibility.json, environment_context.json, archivist utils (for healing), trust_index.yaml

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Bridge integration
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "bridge"))
from bridge_engine import EnhancedBridgeEngine

# CONFIG PATHS
MODEL_CONFIG_PATH  =  Path("/root/config/model_compatibility.json")
TRUST_INDEX_PATH  =  Path("/root/config/trust_index.yaml")
ENV_CONTEXT_PATH  =  Path("/root/config/environment_context.json")
BOOT_LOG_PATH  =  Path("C:/Engineers/root/logs/boot_logs/llm_toggle.log")

# Bridge instance
bridge  =  EnhancedBridgeEngine()

# Active state (stub)
ACTIVE_MODEL  =  None

def log_toggle_event(message):
    """Appends a log entry with timestamp."""
    timestamp  =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(BOOT_LOG_PATH, "a", encoding = "utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def load_model_registry():
    """Loads model compatibility settings."""
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError("Model compatibility config missing.")
    with open(MODEL_CONFIG_PATH, 'r') as f:
        return json.load(f)

def load_env_context():
    """Loads global environment context."""
    if not ENV_CONTEXT_PATH.exists():
        raise FileNotFoundError("Environment context file missing.")
    with open(ENV_CONTEXT_PATH, 'r') as f:
        return json.load(f)

def save_env_context(data):
    """Writes updated environment context."""
    with open(ENV_CONTEXT_PATH, 'w') as f:
        json.dump(data, f, indent = 2)

def select_model(intent, fallback_order = None):
    """
    Selects an LLM based on intent.
    - intent: e.g., 'text2sql', 'summarization', etc.
    - fallback_order: list of preferred model names, optional.
    """
    registry  =  load_model_registry()
    models  =  registry.get(intent)

    if not models:
        raise ValueError(f"No models registered for intent: {intent}")

    if not fallback_order:
        context  =  load_env_context()
        fallback_order  =  context.get("model_fallback_order", [])

    if fallback_order:
        for model in fallback_order:
            if model in models:
                return model
    return models[0]  # Default to first available

def activate_model(model_name):
    """
    Uses the bridge system to send a ping to the model.
    Initializes the model and sets it as active. Updates env context.
    """
    global ACTIVE_MODEL
    test_prompt  =  "You are online. Respond with 'ready'."
    response  =  bridge.send_to_llm(model_name, test_prompt)
    if "ready" in response.lower():
        ACTIVE_MODEL  =  model_name
        log_toggle_event(f"✅ Activated model: {model_name}")

        # Update environment context
        context  =  load_env_context()
        if "active_models" not in context:
            context["active_models"]  =  {}
        context["active_models"]["last"]  =  model_name
        save_env_context(context)

    else:
        raise RuntimeError(f"Model '{model_name}' failed readiness check: {response}")

def toggle_model(intent, fallback_order = None):
    """Wrapper to select and activate model."""
    try:
        model  =  select_model(intent, fallback_order)
        activate_model(model)
    except Exception as e:
        error_msg  =  f"Model toggle failed for intent '{intent}': {str(e)}"
        print(error_msg)
        log_toggle_event(f"❌ {error_msg}")
        # TODO: Fallback to archivist healing

if __name__ == "__main__":
    toggle_model("text2sql")
