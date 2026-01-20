# feedback_handler.py
# Purpose: Receive feedback on model responses, log all data, adjust trust
# Location: /Systems/engine/common/feedback_handler.py

import sqlite3
import yaml
from datetime import datetime
from pathlib import Path

TRUST_PATH = Path("Systems/engine/common/trust_index.yaml")
LOG_PATH = Path("Systems/engine/common/feedback_log.db")
TRUST_PAGE_LIMIT = 20  # number of feedback entries before offloading

# Ensure feedback_log.db is initialized
def ensure_log_schema():
    conn = sqlite3.connect(LOG_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            timestamp TEXT,
            model TEXT,
            prompt TEXT,
            response TEXT,
            correct TEXT,
            was_helpful INTEGER,
            trust_delta REAL
        )
    """)
    conn.commit()
    conn.close()


def load_trust():
    if TRUST_PATH.exists():
        with open(TRUST_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_trust(trust_map):
    with open(TRUST_PATH, "w") as f:
        yaml.dump(trust_map, f)


def adjust_trust(model, was_helpful):
    trust = load_trust()
    current = trust.get(model, 0.75)
    delta = 0.01 if was_helpful else -0.02
    trust[model] = max(0.0, min(1.0, current + delta))
    save_trust(trust)
    return delta


def log_feedback(model, prompt, response, correct, was_helpful):
    ensure_log_schema()
    delta = adjust_trust(model, was_helpful)
    conn = sqlite3.connect(LOG_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)", (
        datetime.utcnow().isoformat(), model, prompt, response, correct, int(was_helpful), delta
    ))
    conn.commit()
    conn.close()

    # Count non-comment lines in YAML
    with open(TRUST_PATH, 'r') as f:
        feedback_lines = [line for line in f if line.strip() and not line.strip().startswith("#")]
    if len(feedback_lines) >= TRUST_PAGE_LIMIT:
        print("[INFO] Paging trust to long-term storage")
        # Insert archiving logic here (e.g., sync to long-term store)
        with open(TRUST_PATH, 'w') as f:
            f.write("# Trust map reset after offload\n")


if __name__ == "__main__":
    log_feedback("Hermes", "Translate: What is 2+2?", "5", "4", was_helpful=False)
    print("Test feedback logged.")
