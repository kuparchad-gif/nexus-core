#!/usr/bin/env python
"""
Central entry point to start the system
- Loads environment variables and model assignments
- Starts watchdog
- Starts each microservice launcher from Systems/roles/
"""

import os
import subprocess
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemLauncher")

# Constants
CONFIG_PATH = os.path.join("Config", "environment_context.json")
WATCHDOG_PATH = os.path.join("boot", "llm_watchdog.py")
SERVICES = [
    "heart",
    "memory",
    "services",
    "bridge",
    "genesis"
]

def load_environment():
    """Load environment variables from config"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            env = json.load(f)
            for key, value in env.items():
                os.environ[key.upper()] = str(value)
        logger.info("Environment context loaded.")
    else:
        logger.warning("Missing environment config. Proceeding with defaults.")

def start_watchdog():
    """Start the watchdog process"""
    if os.path.exists(WATCHDOG_PATH):
        subprocess.Popen(["python", WATCHDOG_PATH])
        logger.info("Watchdog launched.")
    else:
        logger.warning(f"Watchdog script not found at {WATCHDOG_PATH}")

def start_service(role):
    """Start a service using its launch script"""
    launch_script = os.path.join("Systems", "roles", f"launch_{role}.py")
    if os.path.exists(launch_script):
        subprocess.Popen(["python", launch_script])
        logger.info(f"Launched: {role}")
    else:
        logger.warning(f"Launch script missing: {launch_script}")

def main():
    """Main entry point"""
    logger.info("System launch initiated...")
    
    # Load environment variables
    load_environment()
    
    # Start watchdog
    start_watchdog()
    
    # Wait for watchdog to start
    time.sleep(3)
    
    # Start services in order
    for service in SERVICES:
        start_service(service)
        # Add a small delay between service starts to prevent resource contention
        time.sleep(2)
    
    logger.info("All services launched.")

if __name__ == "__main__":
    main()