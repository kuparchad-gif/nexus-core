# swarm_memory.py
# Purpose: Route model outputs to per-model memory and record swarm consensus to a master DB.
# Location: /Systems/memory/swarm_memory.py

import sqlite3
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("C:/Engineers/root/Systems/memory")
MASTER_DB = MEMORY_DIR / "swarm_master.db"


def log_to_model_db(model, prompt, response):
    db_path = MEMORY_DIR / f"{model.replace(' ', '_')}.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            timestamp TEXT,
            prompt TEXT,
            response TEXT
        )
    """)
    c.execute("INSERT INTO memory VALUES (?, ?, ?)", (datetime.utcnow().isoformat(), prompt, response))
    conn.commit()
    conn.close()


def log_to_master_db(intent, consensus, binary_map):
    conn = sqlite3.connect(MASTER_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS swarm_log (
            timestamp TEXT,
            intent TEXT,
            consensus TEXT,
            model TEXT,
            activation INTEGER
        )
    """)
    for model, activation in binary_map:
        c.execute("INSERT INTO swarm_log VALUES (?, ?, ?, ?, ?)",
                  (datetime.utcnow().isoformat(), intent, consensus, model, activation))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Example test run
    log_to_model_db("Qwen 7B", "What is 2+2?", "4")
    log_to_master_db("math", "4", [("Qwen 7B", 1), ("Hermes", 1), ("Phi-4", 0)])
    print("Test entries written.")
