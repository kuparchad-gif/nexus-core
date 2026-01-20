# 📂 Path: /Systems/engine/memory/memory_logger.py

import os
import json
from datetime import datetime

LOG_PATH = '/memory/logs/guardian_memory_log.json'

def log_memory_event(event_type, description, details=None):
    event = {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "type": event_type,
        "description": description,
        "details": details or {}
    }

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(event)

    with open(LOG_PATH, 'w') as f:
        json.dump(logs, f, indent=2)

    print(f"[Memory Logger] Event logged: {event_type}")

if __name__ == "__main__":
    log_memory_event("System", "Guardian Memory Logger Booted Successfully")
