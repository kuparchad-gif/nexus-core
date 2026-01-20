# File: /root/utils/fs_tools.py

# Instructions
# Purpose: Shared utility functions for safe file system access and manipulation
# Usage: Import into any microservice that needs to read, write, move, or delete files
# Examples:
#   - create_dir("logs/planner/")
#   - write_file("data/session.txt", "Hello")
#   - content = read_file("data/config.json")
#   - copy_file("models/base.txt", "models/backup.txt")
# Notes:
# - Automatically logs actions to Archivist via session_manager
# - Respects system structure for traceability and modular autonomy

import os
import shutil
import json
from datetime import datetime
from common.session_manager import append_to_latest_session

def log_fs_action(agent, action, details):
    timestamp = datetime.utcnow().isoformat()
    entry = {
        "timestamp": timestamp,
        "agent": agent,
        "action": action,
        "details": details
    }
    append_to_latest_session("fs_log", entry)

def create_dir(path, agent="fs_tools"):
    os.makedirs(path, exist_ok=True)
    log_fs_action(agent, "create_dir", {"path": path})

def write_file(path, data, agent="fs_tools"):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, indent=2)
        else:
            f.write(str(data))
    log_fs_action(agent, "write_file", {"path": path})

def read_file(path, agent="fs_tools"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    log_fs_action(agent, "read_file", {"path": path})
    return content

def copy_file(src, dest, agent="fs_tools"):
    shutil.copy2(src, dest)
    log_fs_action(agent, "copy_file", {"src": src, "dest": dest})

def move_file(src, dest, agent="fs_tools"):
    shutil.move(src, dest)
    log_fs_action(agent, "move_file", {"src": src, "dest": dest})

def delete_file(path, agent="fs_tools"):
    if os.path.exists(path):
        os.remove(path)
        log_fs_action(agent, "delete_file", {"path": path})

def delete_dir(path, agent="fs_tools"):
    if os.path.exists(path):
        shutil.rmtree(path)
        log_fs_action(agent, "delete_dir", {"path": path})

def list_files(path, agent="fs_tools"):
    if os.path.isdir(path):
        files = os.listdir(path)
        log_fs_action(agent, "list_files", {"path": path, "count": len(files)})
        return files
    return []
