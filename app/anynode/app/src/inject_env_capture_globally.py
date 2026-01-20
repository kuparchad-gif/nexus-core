# File: /root/utils/inject_env_capture_globally.py

# Instructions
# Purpose: Scans all .py files in the project and inserts environment capture lines if missing
# Usage: Run this once after adding session_manager to ensure all services self-report environment
# Examples:
#   - python /root/utils/inject_env_capture_globally.py
# Notes:
# - Adjusts import line for bridge modules using relative paths
# - Skips files that already have capture logic

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["ROOT"] = ROOT

IMPORT_GLOBAL = "from common.session_manager import capture_environment_variables"
CALL_LINE = "capture_environment_variables()"

IMPORT_BRIDGE = "from ..Systems.engine.common.session_manager import capture_environment_variables"

BRIDGE_KEYWORD = os.path.join("bridge").lower()

def inject_capture_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    has_import = any("capture_environment_variables" in line for line in lines)
    if has_import:
        return  # Already injected

    is_bridge = BRIDGE_KEYWORD in file_path.lower()
    import_line = IMPORT_BRIDGE if is_bridge else IMPORT_GLOBAL

    new_lines = [import_line + "\n", CALL_LINE + "\n", "\n"] + lines

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[INJECTED] {file_path}")

def walk_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                full_path = os.path.join(dirpath, file)
                inject_capture_code(full_path)

if __name__ == "__main__":
    walk_directory(os.path.abspath(os.path.join(ROOT, "..")))
