# File: /root/utils/file_mover_and_patcher.py

# Instructions
# Purpose: Moves a given file to a target location and patches all references to it in /scripts and /bridge
# Usage: Run this script manually or from a scheduler after relocating files
# Examples:
#   - python /root/utils/file_mover_and_patcher.py
# Notes:
# - Adjusts import paths dynamically for both global and relative use
# - Ideal for relocating files like session_manager

import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["ROOT"] = ROOT

# Customize these:
SOURCE_FILE = os.path.join(ROOT, "..", "scripts", "session_manager.py")
DEST_FILE = os.path.join(ROOT, "..", "common", "session_manager.py")
PATCH_TARGETS = ["scripts", "bridge"]

IMPORT_LINE = "from common.session_manager import load_sessions, save_session, append_to_latest_session"
IMPORT_LINE_BRIDGE = "from ..Systems.engine.common.session_manager import load_sessions, save_session, append_to_latest_session"


def move_file():
    if os.path.exists(DEST_FILE):
        print("[INFO] Destination file already exists. Skipping move.")
    else:
        shutil.move(SOURCE_FILE, DEST_FILE)
        print(f"[MOVED] {SOURCE_FILE} -> {DEST_FILE}")

def patch_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    patched = False
    for i, line in enumerate(lines):
        if "from" in line and "session_manager" in line:
            if "bridge" in filepath.lower():
                lines[i] = IMPORT_LINE_BRIDGE + "\n"
            else:
                lines[i] = IMPORT_LINE + "\n"
            patched = True

    if patched:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[PATCHED] {filepath}")

def patch_imports():
    project_root = os.path.abspath(os.path.join(ROOT, ".."))
    for target in PATCH_TARGETS:
        search_path = os.path.join(project_root, target)
        for dirpath, _, filenames in os.walk(search_path):
            for file in filenames:
                if file.endswith(".py"):
                    full_path = os.path.join(dirpath, file)
                    patch_file(full_path)

if __name__ == "__main__":
    move_file()
    patch_imports()
