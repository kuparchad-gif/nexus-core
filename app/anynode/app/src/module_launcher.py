#!/usr/bin/env python
"""
Module Launcher for Viren Boot Process

This script launches all required modules during the Viren boot process.
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"module_launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("module_launcher")

# Define modules to launch
MODULES = [
    {
        "name": "Bootstrap Environment",
        "script": "boot/bootstrap_environment.py",
        "required": True,
        "wait": True
    },
    {
        "name": "LLM Loader",
        "script": "boot/llm_loader.py",
        "required": True,
        "wait": True
    },
    {
        "name": "Resource Monitor",
        "script": "boot/resource_monitor.py",
        "required": False,
        "wait": False
    },
    {
        "name": "LLM Watchdog",
        "script": "boot/llm_watchdog.py",
        "required": False,
        "wait": False
    },
    {
        "name": "Viren Sync Initialization",
        "script": "boot/viren_sync_init.py",
        "required": False,
        "wait": True
    }
]

def launch_module(module):
    """Launch a module and return its process."""
    name = module["name"]
    script = module["script"]
    required = module["required"]
    wait = module["wait"]
    
    logger.info(f"Launching module: {name}")
    
    try:
        if wait:
            # Run synchronously
            result = subprocess.run(
                [sys.executable, script],
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Module {name} failed with code {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                
                if required:
                    logger.critical(f"Required module {name} failed, aborting boot process")
                    sys.exit(1)
            else:
                logger.info(f"Module {name} completed successfully")
            
            return None
        else:
            # Run asynchronously
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Module {name} launched with PID {process.pid}")
            return process
    except Exception as e:
        logger.error(f"Failed to launch module {name}: {str(e)}")
        
        if required:
            logger.critical(f"Required module {name} failed to launch, aborting boot process")
            sys.exit(1)
        
        return None

def main():
    """Main entry point."""
    logger.info("Starting module launcher")
    
    # Track async processes
    processes = []
    
    # Launch modules
    for module in MODULES:
        process = launch_module(module)
        if process:
            processes.append((module["name"], process))
    
    # Wait for async processes to complete (optional)
    # for name, process in processes:
    #     process.wait()
    #     logger.info(f"Module {name} completed with code {process.returncode}")
    
    logger.info("All modules launched successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())