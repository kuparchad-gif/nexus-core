# 📂 Path: /Utilities/guardian_core/guardian_service.py

import time
import threading
import json
import os

GUARDIAN_LOG_FILE = '/Memory/streams/guardian_watchlog.json'
PLANNER_MEMORY_FILE = '/Memory/streams/eden_memory_map.json'

class GuardianService:
    def __init__(self, drones):
        self.drones = drones  # List of DroneEntity objects
        self.watch_log = []
        self.load_existing_log()

    def load_existing_log(self):
        if os.path.exists(GUARDIAN_LOG_FILE):
            with open(GUARDIAN_LOG_FILE, 'r') as f:
                self.watch_log = json.load(f)
            print("[Guardian] Existing Watchlog loaded.")
        else:
            self.watch_log = []
            print("[Guardian] New Watchlog initialized.")

    def scan_fleet(self):
        health_report = []
        for drone in self.drones:
            status = drone.report_status()
            health_report.append(status)
        return health_report

    def scan_memory_integrity(self):
        if not os.path.exists(PLANNER_MEMORY_FILE):
            print("[Guardian] EdenMemory map missing!")
            return None
        with open(PLANNER_MEMORY_FILE, 'r') as f:
            memory_map = json.load(f)
        return {
            "total_shards": len(memory_map),
            "memory_integrity": "healthy" if len(memory_map) > 0 else "warning: no memory!"
        }

    def log_health_snapshot(self):
        snapshot = {
            "timestamp": time.time(),
            "fleet_status": self.scan_fleet(),
            "memory_status": self.scan_memory_integrity()
        }
        self.watch_log.append(snapshot)
        self.save_log()
        print(f"[Guardian] Health snapshot logged at {snapshot['timestamp']}.")

    def save_log(self):
        with open(GUARDIAN_LOG_FILE, 'w') as f:
            json.dump(self.watch_log, f, indent=2)
        print("[Guardian] Watchlog saved.")

    def start_guardian_cycle(self, interval_seconds=104):
        def guardian_loop():
            while True:
                self.log_health_snapshot()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=guardian_loop, daemon=True)
        thread.start()

# Example Usage:
# guardian = GuardianService(drones=[golden, whisper, vault])
# guardian.start_guardian_cycle()
