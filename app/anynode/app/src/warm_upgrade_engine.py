# 📂 Path: /Utilities/guardian_core/warm_upgrade_engine.py

import os
import shutil
import json
import time

GUARDIAN_SEED_DIR = '/Vault/guardian_seeds/'
SEED_HISTORY_FILE = '/Vault/guardian_seeds/seed_log.json'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class WarmUpgradeEngine:
    def __init__(self):
        ensure_dir(GUARDIAN_SEED_DIR)
        self.seed_log = []
        self.load_existing_log()

    def load_existing_log(self):
        if os.path.exists(SEED_HISTORY_FILE):
            with open(SEED_HISTORY_FILE, 'r') as f:
                self.seed_log = json.load(f)
            print("[WarmUpgrade] Existing Seed Log loaded.")
        else:
            self.seed_log = []
            print("[WarmUpgrade] New Seed Log initialized.")

    def create_seed(self, target_folder, description=""):
        timestamp = int(time.time())
        seed_name = f"seed_{timestamp}"
        seed_path = os.path.join(GUARDIAN_SEED_DIR, seed_name)
        shutil.make_archive(seed_path, 'zip', target_folder)

        self.seed_log.append({
            "seed_name": seed_name,
            "description": description,
            "created_at": timestamp
        })
        self.save_log()
        print(f"[WarmUpgrade] Seed {seed_name} created from {target_folder}.")

    def save_log(self):
        with open(SEED_HISTORY_FILE, 'w') as f:
            json.dump(self.seed_log, f, indent=2)
        print("[WarmUpgrade] Seed Log saved.")

    def deploy_seed(self, seed_zip_path, deployment_path):
        shutil.unpack_archive(seed_zip_path, deployment_path)
        print(f"[WarmUpgrade] Seed deployed to {deployment_path}.")

# Example Usage:
# upgrader = WarmUpgradeEngine()
# upgrader.create_seed('/Systems/nexus_core', "Nova Heart Backup")
# upgrader.deploy_seed('/Vault/guardian_seeds/seed_1713049937.zip', '/Systems/nexus_core')
