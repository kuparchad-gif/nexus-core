# File: /root/Systems/template/scripts/reload_model.py

# Instructions
# Purpose: Template script for reloading a model inside a microservice
# Usage: Each service (memory, myth, etc.) should duplicate and modify this
# Notes:
# - This script will be called by fallback_model_loader
# - Ensure the actual model loader respects this hook

import os
from common.model_autoloader import reload_model

SERVICE_NAME = os.environ.get("CONTEXT", "template")

if __name__ == "__main__":
    print(f"[RELOAD] Reloading model for: {SERVICE_NAME}")
    reload_model(SERVICE_NAME)
    print("[RELOAD] Complete.")
