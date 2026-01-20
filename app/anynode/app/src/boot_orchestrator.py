# File: /root/utils/boot_orchestrator.py

# Instructions
# Purpose: Master boot logic that launches service discovery and role offloading together
# Usage: Called by launchers or manually during hot start of a node
# Examples:
#   - python /root/utils/boot_orchestrator.py
# Notes:
# - Requires CONTEXT environment variable to be set on each node

import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["ROOT"] = ROOT

CONTEXT = os.environ.get("CONTEXT", "undefined")

scripts = [
    "service_discovery.py",
    "role_offloader.py"
]

print(f"[BOOT] Starting orchestration for context: {CONTEXT}")

for script in scripts:
    path = os.path.join(ROOT, script)
    if os.path.exists(path):
        subprocess.run(["python", path])
    else:
        print(f"[MISSING] {script} not found")
