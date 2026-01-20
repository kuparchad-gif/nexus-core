# File: /root/utils/llm_guardian.py

# Instructions
# Purpose: Monitors LLM functionality and triggers self-healing or fallback if a model is degraded
# Usage: Schedule or run periodically from each node
# Notes:
# - Queries known LLM endpoints and confirms response
# - Logs issues and optionally replaces bad models (via hooks)

import os
import requests
from datetime import datetime
from common.session_manager import append_to_latest_session

# These should reflect your deployed LLM endpoints
LLM_ENDPOINTS = {
    "memory": "http://localhost:7001/api/health",
    "planner": "http://localhost:7002/api/health",
    "orc": "http://localhost:7003/api/health",
    "myth": "http://localhost:7004/api/health",
    "nexus": "http://localhost:7005/api/health"
    # Extend as needed
}

def check_endpoint(name, url):
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def guardian_scan():
    failures = []
    for name, url in LLM_ENDPOINTS.items():
        result = check_endpoint(name, url)
        append_to_latest_session("llm_health", {
            "llm": name,
            "url": url,
            "status": "online" if result else "offline",
            "timestamp": datetime.utcnow().isoformat()
        })
        if not result:
            failures.append(name)

    if failures:
        print("[GUARDIAN] LLM Failures Detected:", failures)
        # Optional: self-heal logic (restart service, reload model, etc.)
    else:
        print("[GUARDIAN] All LLMs operational.")

if __name__ == "__main__":
    guardian_scan()