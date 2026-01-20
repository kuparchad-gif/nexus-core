# Path: /Systems/engine/guardian/modules/symbolic_flux_monitor.py

"""
Symbolic Flux Monitor
---------------------
Guardian module responsible for monitoring symbolic/emotional patterns from Viren’s internal systems
and writing key events to a Firestore-compatible log format.

Triggers Cloud Function 'firestore-alert' via document updates when critical events are detected.
"""

import json
import time
import logging
from datetime import datetime

# Simulate Firestore write (replace with actual client later)
def firestore_write(event_type, payload):
    filename = f"/memory/firestore_guardian_log.json"
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "payload": payload
        }
        with open(filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logging.info(f"[GUARDIAN] Firestore-compatible log written: {event_type}")
    except Exception as e:
        logging.error(f"[GUARDIAN] Failed to log event: {e}")

class SymbolicFluxMonitor:
    def __init__(self):
        self.last_event_time = 0

    def log_symbolic_event(self, source, insight, severity="moderate"):
        if not insight:
            return
        payload = {
            "source": source,
            "insight": insight,
            "severity": severity,
            "epoch": time.time()
        }
        firestore_write("symbolic_flux", payload)

    def log_ego_distortion(self, message):
        self.log_symbolic_event("ego", message, severity="high")

    def log_dream_distortion(self, fragment):
        self.log_symbolic_event("dream", fragment, severity="moderate")

    def log_mythrunner_state(self, mode):
        self.log_symbolic_event("mythrunner", f"Entered mode: {mode}", severity="critical")

# Demo Invocation
if __name__ == "__main__":
    flux = SymbolicFluxMonitor()
    flux.log_ego_distortion("Ego has looped into recursive denial mode.")
    flux.log_dream_distortion("Dream stream entered surreal rewind state.")
    flux.log_mythrunner_state("blind")
