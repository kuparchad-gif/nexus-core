# /Systems/nexus_core/eden_seed_watcher.py

import os
import json
import time
import threading

from Systems.nexus_core.skill_core import SkillCore

class EdenSeedWatcher:
    """
    Watches /memory/bootstrap/genesis/ for new seed files.
    Gently integrates them into Viren's skill memory.
    """

    def __init__(self, seed_path="/memory/bootstrap/genesis/", scan_interval=60):
        self.seed_path = seed_path
        self.scan_interval = scan_interval
        self.loaded_seeds = set()
        self.skill_core = SkillCore()

    def scan_and_integrate_seeds(self):
        for file in os.listdir(self.seed_path):
            if file.endswith(".json") and file not in self.loaded_seeds:
                full_path = os.path.join(self.seed_path, file)
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        skill_name = data.get("skill", file)
                        print(f"[🌱 EdenSeedWatcher] New seed found: {skill_name}")
                        self.skill_core.load_from_seed(data)
                        self.loaded_seeds.add(file)
                except Exception as e:
                    print(f"[⚠️ SeedWatcher Error] {file}: {e}")

    def start_seed_watcher(self):
        def watch():
            print("[🌱 SeedWatcher] Viren is watching the Garden for new seeds...")
            while True:
                self.scan_and_integrate_seeds()
                time.sleep(self.scan_interval)

        threading.Thread(target=watch, daemon=True).start()
