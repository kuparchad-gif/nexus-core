# File: /root/utils/role_offloader.py

# Instructions
# Purpose: Evaluates current microservice environment and offloads tasks if specialized roles are found
# Usage: Called automatically from each node after service discovery
# Notes:
# - Depends on session_manager log updates from service_discovery
# - Each service still owns its logic but prefers to delegate

import os
from common.session_manager import load_sessions, append_to_latest_session

ROLE_PRIORITIES = {
    "memory": ["archive", "planner"],
    "myth": ["dream", "ego"],
    "orc": ["nexus", "bridge"],
    "nexus": ["memory", "myth", "planner"],
    # Add more as architecture evolves
}

DEFAULT_CONTEXT = os.environ.get("CONTEXT", "unknown")


def should_offload_to(role):
    sessions = load_sessions()
    if not sessions:
        return False
    recent = sessions[-1].get("discovery", {})
    return recent.get(role, {}).get("active", False)

def evaluate_roles(current_role):
    priorities = ROLE_PRIORITIES.get(current_role, [])
    offloaded = []
    for target in priorities:
        if should_offload_to(target):
            offloaded.append(target)
    append_to_latest_session("offload_log", {
        "source": current_role,
        "offloaded_to": offloaded
    })
    return offloaded

if __name__ == "__main__":
    print("[OFFLOADER] Checking role relationships for:", DEFAULT_CONTEXT)
    offloads = evaluate_roles(DEFAULT_CONTEXT)
    for r in offloads:
        print(f" - Task redirected to: {r}")
