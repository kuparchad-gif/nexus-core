# File: root/scripts/knowledge_loader.py
# Purpose: Scan full engineering system, build file/dir map for engineer awareness

import os

def build_filesystem_map(base_dir):
    fs_map = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            path = os.path.relpath(os.path.join(root, file), base_dir)
            fs_map.append(path)
    return fs_map
