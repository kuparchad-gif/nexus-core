# File: /root/utils/boot_status_validator.py

# Instructions
# Purpose: Verify whether key modules and subsystems are available and importable
# Usage: Run as a health check or at boot to confirm the runtime is complete
# Examples:
#   - python /root/utils/boot_status_validator.py
# Notes:
# - Useful for verifying structural integrity across updates or patches
# - Add additional targets as modules are expanded

import importlib
import sys

MODULES_TO_CHECK = [
    "common.session_manager",
    "common.fs_tools",
    "utils.inject_env_capture_globally",
    "utils.file_mover_and_patcher"
]

def check_module(name):
    try:
        importlib.import_module(name)
        print(f"[OK] Module '{name}' is importable.")
    except Exception as e:
        print(f"[FAIL] Module '{name}' failed: {e}")

if __name__ == "__main__":
    for module in MODULES_TO_CHECK:
        check_module(module)
