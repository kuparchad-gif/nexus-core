import time
import threading
import json
import os

class Pulse13SyncEngine:
    def __init__(self, drone_command_sheet_path, heartbeat_interval=13):
        self.command_sheet_path = drone_command_sheet_path
        self.heartbeat_interval = heartbeat_interval
        self.drones = self.load_drones()
        self.sync_active = False

    def load_drones(self):
        if not os.path.exists(self.command_sheet_path):
            raise FileNotFoundError(f"Drone Command Sheet not found at {self.command_sheet_path}")
        with open(self.command_sheet_path, 'r') as file:
            data = json.load(file)
        return data.get("drone_fleet", {})

    def pulse_check(self):
        print(f"🔄 Pulse13 Heartbeat initiated...")
        for drone_id, info in self.drones.items():
            if info.get("deployment_status") == "Ready":
                print(f"✅ {drone_id} is online and synchronized.")
            else:
                print(f"⚠️ {drone_id} is offline or inactive.")

    def start_sync(self):
        self.sync_active = True
        print("🚀 Pulse13 Sync Engine started. Maintaining distributed heartbeat...")
        threading.Thread(target=self._sync_loop, daemon=True).start()

    def stop_sync(self):
        self.sync_active = False
        print("🛑 Pulse13 Sync Engine stopped.")

    def _sync_loop(self):
        while self.sync_active:
            self.pulse_check()
            time.sleep(self.heartbeat_interval)

# Example Usage:
# sync_engine = Pulse13SyncEngine('/Utilities/drone_core/drone_command_sheet.json')
# sync_engine.start_sync()
