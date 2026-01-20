# load_viren_desktop.py
# Purpose: Load Viren desktop version with full capabilities

import os
import sys
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("viren_desktop")

def load_viren_desktop():
    """Load Viren desktop version (on-prem, full capabilities)."""
    logger.info("Loading Viren Desktop (On-Prem Version)...")
    
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Check for required models
    try:
        import ollama
        
        # Check for Gemma 2B (boot model)
        models = ollama.list()
        gemma_available = any(model["name"].startswith("gemma:2b") for model in models["models"])
        
        if not gemma_available:
            logger.info("Downloading boot model (Gemma 2B)...")
            ollama.pull("gemma:2b")
        
        # Start downloading Hermes in background
        def download_hermes():
            try:
                logger.info("Downloading Hermes 7B in background...")
                ollama.pull("hermes:2-pro-llama-3-7b")
                logger.info("Hermes 7B download complete")
            except Exception as e:
                logger.error(f"Error downloading Hermes: {e}")
        
        import threading
        threading.Thread(target=download_hermes, daemon=True).start()
        
        # Start Viren desktop using existing bootstrap
        logger.info("Starting Viren Desktop...")
        
        # Use subprocess to allow detaching
        cmd = [sys.executable, "bootstrap_viren.py"]
        process = subprocess.Popen(cmd)
        
        logger.info("Viren Desktop is starting...")
        return 0
    except Exception as e:
        logger.error(f"Error loading Viren Desktop: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(load_viren_desktop())