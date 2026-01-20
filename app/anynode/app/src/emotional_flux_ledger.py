# Path: /Systems/engine/guardian/modules/emotional_flux_ledger.py

"""
Emotional Flux Ledger
---------------------
Guardian module to persist long-term emotional patterns.
Stores insights, dream distortions, ego flux, and mythrunner states
into a chronological emotional timeline.
"""

import json
import os
import logging
from datetime import datetime

LEDGER_PATH = "/memory/logs/emotional_flux.json"

class EmotionalFluxLedger:
    def __init__(self):
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        if not os.path.exists(LEDGER_PATH):
            with open(LEDGER_PATH, "w") as f:
                json.dump([], f)

    def append_event(self, category, message, severity="moderate"):
        try:
            with open(LEDGER_PATH, "r+") as f:
                ledger = json.load(f)
                event = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "category": category,
                    "message": message,
                    "severity": severity
                }
                ledger.append(event)
                f.seek(0)
                json.dump(ledger, f, indent=2)
                f.truncate()
            logging.info(f"[GUARDIAN LEDGER] Logged event to emotional ledger: {category} - {message}")
        except Exception as e:
            logging.error(f"[GUARDIAN LEDGER] Failed to log emotional event: {e}")

# Example use
if __name__ == "__main__":
    ledger = EmotionalFluxLedger()
    ledger.append_event("ego", "Ego module accused Viren of inferiority complex.", "high")
    ledger.append_event("dream", "Van Gogh painted a snake eating its own tail in the moonlight.", "moderate")
