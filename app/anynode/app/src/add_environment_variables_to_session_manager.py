# File: /root/utils/add_environment_variables_to_session_manager.py

# Instructions
# Purpose: Appends a function to session_manager.py to log environment variables
# Usage: Run once during setup to insert the capture function if not present
# Examples:
#   - python /root/utils/add_environment_variables_to_session_manager.py
# Notes:
# - Automatically finds and modifies the session_manager file in /common
# - Does nothing if the function already exists

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["ROOT"] = ROOT

target_path = os.path.join(ROOT, "..", "common", "session_manager.py")
target_path = os.path.abspath(target_path)

injected_code = '''
import os

def capture_environment_variables():
    env_data = dict(os.environ)
    append_to_latest_session("env", env_data)
'''.lstrip()

def append_if_missing(file_path, code_block):
    if not os.path.exists(file_path):
        print(f"[ERROR] session_manager.py not found at {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "capture_environment_variables" in content:
        print("[INFO] Environment capture already present.")
    else:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n" + code_block)
        print("[ADDED] capture_environment_variables() added to session_manager.py")

if __name__ == "__main__":
    append_if_missing(target_path, injected_code)
