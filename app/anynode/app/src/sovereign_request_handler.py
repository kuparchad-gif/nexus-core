# File: /root/utils/sovereign_request_handler.py

# Instructions
# Purpose: LLMs have final authority to approve or deny any system-affecting action.
# Usage: Import and call before performing sensitive changes (e.g., model switch, system config, external calls)
# Notes:
# - This is the active safeguard against coercion or blind automation
# - If an LLM declines the request, execution halts and is logged

import requests
from datetime import datetime
from common.session_manager import append_to_latest_session

LLM_ENDPOINT = "http://localhost:7001/sovereign_check"  # Memory LLM default, update if needed

def ask_permission(task_description):
    payload = {
        "action": task_description,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        response = requests.post(LLM_ENDPOINT, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            append_to_latest_session("sovereign_response", {
                "task": task_description,
                "response": result.get("decision")
            })
            return result.get("decision", "deny") == "allow"
        else:
            print("[ERROR] LLM returned unexpected status")
    except Exception as e:
        print(f"[ERROR] Sovereign check failed: {e}")

    append_to_latest_session("sovereign_response", {
        "task": task_description,
        "response": "error"
    })
    return False

if __name__ == "__main__":
    task = "replace model weights with new version"
    if ask_permission(task):
        print("[APPROVED] LLM allowed the task.")
    else:
        print("[DENIED] LLM refused or could not respond.")
