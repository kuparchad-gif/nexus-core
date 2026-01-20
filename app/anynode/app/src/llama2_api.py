# /eden_engineering/scripts/llama2_api.py
# Purpose: Unified local API interface for managing and querying all LLM Engineers running via LM Studio or similar local backend.
# Handles model selection, routing, memory/context injection, logging, error handling.
# NOT for Baidu/Wenxin/external cloud APIs—this is purely for your on-prem, multi-agent engineering system.

import requests
import os
import json
from pathlib import Path

# Config
LMSTUDIO_URL = "http://127.0.0.1:1313/v1"  # Match LM Studio API endpoint/port
SESSIONS_DIR = Path(__file__).parent.parent / "memory" / "sessions"
CORPUS_DIR = Path(__file__).parent.parent / "memory" / "training_corpus"
LOG_FILE = Path(__file__).parent.parent / "logs" / "system_errors" / "llama2_api.log"

def log_event(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def list_models(max_params=12):
    """List all available models <=12B params"""
    try:
        resp = requests.get(f"{LMSTUDIO_URL}/models")
        resp.raise_for_status()
        models = resp.json()["data"]
        return [m for m in models if m["type"] == "llm" and ("b" in str(m.get("id","")).lower() and _parse_params(m) <= max_params)]
    except Exception as e:
        log_event(f"Model listing error: {e}")
        return []

def _parse_params(model):
    # Extract params count from model metadata or ID (e.g., "8b", "12b")
    for key in ["id", "name"]:
        if key in model:
            try:
                return int([x for x in model[key].lower().split("-") if "b" in x][0].replace("b", ""))
            except:
                continue
    return 0

def load_model(model_id):
    """Load a model by ID if not already loaded"""
    try:
        requests.post(f"{LMSTUDIO_URL}/models/{model_id}/load")
    except Exception as e:
        log_event(f"Error loading model {model_id}: {e}")

def unload_model(model_id):
    try:
        requests.post(f"{LMSTUDIO_URL}/models/{model_id}/unload")
    except Exception as e:
        log_event(f"Error unloading model {model_id}: {e}")

def send_prompt(model_id, prompt, history=None, extra_context=None):
    """Send a prompt to a specific model, adding session/corpus/history context as needed"""
    payload = {
        "model": model_id,
        "prompt": prompt,
        "history": history or [],
        "extra_context": extra_context or "",
    }
    try:
        r = requests.post(f"{LMSTUDIO_URL}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log_event(f"Prompt error ({model_id}): {e}")
        return {"error": str(e)}

def get_memory(agent_name):
    """Load all sessions and corpus data for a given agent (Engineer)"""
    sessions = []
    if SESSIONS_DIR.exists():
        for f in SESSIONS_DIR.glob("*.json"):
            with open(f, "r", encoding="utf-8") as file:
                sessions.extend(json.load(file))
    corpus = []
    if CORPUS_DIR.exists():
        for f in CORPUS_DIR.glob("*.*"):
            with open(f, "r", encoding="utf-8") as file:
                corpus.append(file.read())
    return {"sessions": sessions, "corpus": corpus}

def engineer_chat(prompt, agent="collective"):
    """Main entry to query the engineers (collective or specific)"""
    available = list_models()
    if not available:
        return {"error": "No suitable LLM models found."}
    # If "collective", pick a consensus (e.g. majority or first loaded)
    # If individual, pick by name or ID
    model_id = available[0]["id"] if agent == "collective" else agent
    memory = get_memory(agent)
    # Build full context: join sessions, corpus, prompt
    context = "\n".join(memory["corpus"]) + "\n" + "\n".join([s["message"] for s in memory["sessions"] if "message" in s])
    return send_prompt(model_id, prompt, extra_context=context)

if __name__ == "__main__":
    # CLI for quick test
    import sys
    prompt = " ".join(sys.argv[1:]) or "Who are you?"
    print(engineer_chat(prompt))
