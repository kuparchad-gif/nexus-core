# File: root/scripts/program_launcher.py
# Purpose: Main orchestrator for Engineer collective

import os
import subprocess
import threading
import sys
import time
from knowledge_loader import build_filesystem_map
from corpus_ingest import ingest_training_corpus
from session_manager import load_sessions, save_session
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
TRAINING_CORPUS = os.path.join(MEMORY_DIR, "training_corpus")
SESSIONS_DIR = os.path.join(MEMORY_DIR, "sessions")
DB_PATH = os.path.join(MEMORY_DIR, "engineers.db")

def launch_llm_agents():
    from models.model_manifest import get_model_list
    models = get_model_list(max_params=12)
    agents = []
    for model in models:
        proc = subprocess.Popen(
            [sys.executable, "llama2_api.py", "--model", model['id']],
            cwd=os.path.join(BASE_DIR, "scripts")
        )
        agents.append(proc)
    return agents

def launch_console():
    console_path = os.path.join(BASE_DIR, "templates", "console.html")
    subprocess.Popen(["python", "-m", "http.server", "8080"], cwd=os.path.dirname(console_path))

def initialize_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (session_id TEXT PRIMARY KEY, data TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS file_index
                     (path TEXT PRIMARY KEY, last_modified DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS knowledge
                     (id INTEGER PRIMARY KEY, content TEXT)''')
        conn.commit()
        conn.close()

if __name__ == "__main__":
    initialize_db()
    print("Launching Engineers...")

    # Step 1: Build map of entire filesystem for engineers' awareness
    fs_map = build_filesystem_map(BASE_DIR)

    # Step 2: Ingest training corpus and inject into all LLMs
    training_data = ingest_training_corpus(TRAINING_CORPUS)

    # Step 3: Load all session history
    sessions = load_sessions(SESSIONS_DIR)

    # Step 4: Launch all LLM agents <= 12B params
    agents = launch_llm_agents()

    # Step 5: Launch the console UI as a subprocess
    launch_console()

    # Step 6: Main thread monitors subprocesses
    try:
        while True:
            for agent in agents:
                if agent.poll() is not None:
                    print("An agent has crashed! Relaunching...")
                    # Logic to relaunch if needed
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down...")
        for agent in agents:
            agent.terminate()
