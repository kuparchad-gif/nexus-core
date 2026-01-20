# File: /root/utils/vector_store_manager.py

# Instructions
# Purpose: Automatically selects the optimal vector database engine (Pinecone, FAISS, Ray)
# Usage: Called during memory or embedding initialization
# Notes:
# - Honors override config if present
# - Validates credentials and local installation
# - Future support planned: Weaviate, Chroma, LanceDB

import os
import json
import importlib.util
from utils.sovereign_request_handler import ask_permission
from common.session_manager import append_to_latest_session

CONFIG_PATH = os.path.join("config", "vector_db_config.json")

VECTOR_ENGINES = ["pinecone", "faiss", "ray"]


def is_module_installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def detect_installed():
    installed = {
        "pinecone": is_module_installed("pinecone") and bool(os.environ.get("PINECONE_API_KEY")),
        "faiss": is_module_installed("faiss") or is_module_installed("faiss_cpu"),
        "ray": is_module_installed("ray")
    }
    return installed


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_config(runtime):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump({"selected_vector_engine": runtime}, f)


def select_best_vector_engine():
    installed = detect_installed()
    if installed.get("pinecone"):
        return "pinecone"
    elif installed.get("faiss"):
        return "faiss"
    elif installed.get("ray"):
        return "ray"
    return "none"


def choose_vector_engine():
    config = load_config()
    override = config.get("selected_vector_engine")

    if override:
        approved = ask_permission(f"Use manually set vector engine: {override}")
        if approved:
            print(f"[VECTOR] Using override: {override}")
            return override

    engine = select_best_vector_engine()
    save_config(engine)
    append_to_latest_session("vector_engine_choice", {"engine": engine})
    print(f"[VECTOR] Auto-selected: {engine}")
    return engine


if __name__ == "__main__":
    choose_vector_engine()
