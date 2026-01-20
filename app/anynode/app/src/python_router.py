# python_router.py
# Purpose: Expose backend functions to frontend or CLI through a unified command interface

import logging
from bridge.model_fetcher import fetch_running_models
from bootstrap_environment import collect_environment

logger = logging.getLogger("python_router")
logging.basicConfig(level=logging.INFO)

def handle_command(command: str, args=None):
    if args is None:
        args = []

    if command == "refresh_models":
        return fetch_running_models()

    elif command == "get_environment":
        return collect_environment()

    else:
        logger.warning(f"[PythonRouter] Unknown command: {command}")
        return {"error": f"Unknown command: {command}"}
