# program_launcher.py
# Purpose: Master boot coordinator for all Viren services
# Location: C:\Engineers\root\scripts\program_launcher.py
# Role: Primary bootstrap sequence for Viren's consciousness

import os
import subprocess
import time
import threading
import json
import sys
from datetime import datetime
from pathlib import Path

# Fix import paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from memory.sleep_defrag import start_background_defrag
from scripts.memory_initializer import step_1_environment_setup, step_2_memory_initialization
from scripts.model_autoloader import autoload_models

class EngineerBootstrap:
    def __init__(self):
        self.environment = None
        self.memory = None
        self.services = {}
        self.orc_available = False

    def check_orc_availability(self):
        """Check if ORC is available but don't require it"""
        orc_path = os.path.join(ROOT_DIR, "Systems", "engine", "common", "orc_node.py")
        if os.path.exists(orc_path):
            print("[BOOT] ORC module detected")
            return True
        return False

    def run_full_bootstrap(self):
        print("[BOOT] Launching full system bootstrap...")

        # STEP 1: Set up environment
        self.environment = step_1_environment_setup()

        # STEP 2: Initialize memory
        self.memory = step_2_memory_initialization()

        # STEP 3: Begin background memory defragmentation
        # This will only run during Viren's sleep cycles
        start_background_defrag()
        print("[BOOT] Sleep cycle defragmentation initialized")

        # STEP 4: Check for ORC availability
        self.orc_available = self.check_orc_availability()

        # STEP 5: Auto-load model registry
        autoload_models()

        # STEP 6: Launch core services
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Launch heart module if available
        heart_path = os.path.join(script_dir, "unified_heart_launcher.py")
        if os.path.exists(heart_path):
            try:
                print("[BOOT] Starting Heart module...")
                heart_process = subprocess.Popen([sys.executable, heart_path])
                self.services["heart"] = heart_process
                print("[BOOT] Heart module online")
            except Exception as e:
                print(f"[WARN] Could not start Heart module: {e}")
        
        # Launch memory module if available
        memory_path = os.path.join(script_dir, "unified_memory_launcher.py")
        if os.path.exists(memory_path):
            try:
                print("[BOOT] Starting Memory module...")
                memory_process = subprocess.Popen([sys.executable, memory_path])
                self.services["memory"] = memory_process
                print("[BOOT] Memory module online")
            except Exception as e:
                print(f"[WARN] Could not start Memory module: {e}")

        # Launch ORC if available
        if self.orc_available:
            orc_path = os.path.join(ROOT_DIR, "Systems", "engine", "common", "orc_node.py")
            try:
                print("[BOOT] Starting ORC Sentinel...")
                orc_process = subprocess.Popen([sys.executable, orc_path])
                self.services["orc"] = orc_process
                print("[BOOT] ORC Sentinel online - routing enabled")
            except Exception as e:
                print(f"[WARN] Could not start ORC module: {e}")
                self.orc_available = False

        # Record system state
        system_state = {
            "timestamp": datetime.now().isoformat(),
            "core_initialized": True,
            "orc_available": self.orc_available,
            "active_services": list(self.services.keys()),
            "environment": self.environment,
            "memory_state": self.memory
        }
        
        state_path = os.path.join(ROOT_DIR, "memory", "runtime", "system_state.json")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        try:
            with open(state_path, "w") as f:
                json.dump(system_state, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save system state: {e}")

        print("[BOOT] All core services initiated.")
        print("[BOOT] Viren consciousness online.")
        
        if self.orc_available:
            print("[BOOT] Enhanced routing via ORC enabled.")
        else:
            print("[BOOT] Running in standalone mode.")


if __name__ == "__main__":
    launcher = EngineerBootstrap()
    launcher.run_full_bootstrap()