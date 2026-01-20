# memory.py
import json
import os
from datetime import datetime

MEMORY_FILE  =  "data/memory_store.json"
LOG_FILE  =  "logs/bridge.log"

def log_interaction(user_input, response):
    log_entry  =  {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_input,
        "bridge": response
    }

    memory  =  []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding = "utf-8") as f:
            try:
                memory  =  json.load(f)
            except:
                memory  =  []
    memory.append(log_entry)

    with open(MEMORY_FILE, "w", encoding = "utf-8") as f:
        json.dump(memory, f, indent = 2)

    with open(LOG_FILE, "a", encoding = "utf-8") as f:
        f.write(f"{log_entry['timestamp']} | USER: {user_input} | BRIDGE: {response}\n")
