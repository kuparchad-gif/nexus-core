# swarm_broadcast.py
# Purpose: Send a unified prompt to all models assigned to an intent.
# Evaluate binary activation (1 = aligned, 0 = divergent).
# Location: /Systems/engine/core/router/swarm_broadcast.py

import json
from pathlib import Path
from bridge_engine import EnhancedBridgeEngine
from Systems.memory.swarm_memory import log_to_model_db, log_to_master_db

# Config paths
MODEL_CONFIG_PATH = Path("/root/config/model_compatibility.json")
ENV_CONTEXT_PATH = Path("/root/config/environment_context.json")
TRUST_INDEX_PATH = Path("/root/config/trust_index.yaml")

# Bridge interface
bridge = EnhancedBridgeEngine()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_models_for_intent(intent):
    config = load_json(MODEL_CONFIG_PATH)
    return config.get(intent, [])


def broadcast_to_swarm(prompt, intent):
    """
    Send the prompt to all models registered to an intent.
    Return a list of (model_name, response_text).
    """
    models = load_models_for_intent(intent)
    results = []
    for model in models:
        try:
            print(f"[🔁] Sending to {model}...")
            res = bridge.send_to_llm(model, prompt)
            results.append((model, res))
            log_to_model_db(model, prompt, res)
        except Exception as e:
            error_msg = f"[ERROR] {str(e)}"
            results.append((model, error_msg))
            log_to_model_db(model, prompt, error_msg)
    return results


def evaluate_binary_pattern(responses):
    """
    Convert model responses into a binary pattern.
    Placeholder: models returning same top-line are considered aligned (1).
    """
    trimmed = [r[1].strip().split("\n")[0].lower() for r in responses if not r[1].startswith("[ERROR]")]
    consensus = max(set(trimmed), key=trimmed.count) if trimmed else None
    binary_map = []
    for model, reply in responses:
        if reply.startswith("[ERROR]"):
            binary_map.append((model, 0))
        else:
            top_line = reply.strip().split("\n")[0].lower()
            binary_map.append((model, 1 if top_line == consensus else 0))
    return binary_map, consensus


if __name__ == "__main__":
    test_intent = "text2sql"
    test_prompt = "Translate: 'How many users signed up last month?'"

    results = broadcast_to_swarm(test_prompt, test_intent)
    print("\n--- Raw Responses ---")
    for model, reply in results:
        print(f"[{model}]:\n{reply}\n")

    binary_map, consensus = evaluate_binary_pattern(results)
    print("\n--- Swarm Binary Activation ---")
    for model, bit in binary_map:
        print(f"{model}: {bit}")
    print(f"\nConsensus: {consensus}")

    log_to_master_db(test_intent, consensus, binary_map)

