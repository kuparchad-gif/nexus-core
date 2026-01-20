#!/usr/bin/env python3
"""
Chunk COMPLETE_FILE_AUDIT.json into 18 TXT docs (100 files/chunk).
Fixed: Handle duplicates as list, not dict.
Run: python chunk_audit.py
Outputs: chunk_1.txt to chunk_18.txt
"""

import json
from pathlib import Path

AUDIT_PATH = Path("COMPLETE_FILE_AUDIT.json")
CHUNK_SIZE = 100
NUM_CHUNKS = 18  # ~1876 / 100 = 18.76

def flatten_files(data):
    """Extract unique files from directories + duplicates (list format)."""
    files = []
    # From directories["."]["files"]
    if "." in data.get("directories", {}):
        files.extend(data["directories"]["."]["files"])
    # From duplicates: List of dicts—expand locations
    duplicates = data.get("duplicates", [])
    for dupe in duplicates:  # Fix: Iterate list
        filename = dupe.get("filename", "unknown")
        locs = dupe.get("locations", [])
        for loc in locs:
            full_entry = loc.copy()
            full_entry["name"] = filename
            files.append(full_entry)
    # Dedupe by full_path
    unique = {f["full_path"]: f for f in files if "full_path" in f}
    return list(unique.values())

def chunk_and_save(files):
    """Split into 18 chunks, save as JSON TXT."""
    total = len(files)
    print(f"Flattened {total} unique files.")
    for i in range(NUM_CHUNKS):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)
        chunk = files[start:end]
        chunk_data = {"chunk": f"{i+1}/{NUM_CHUNKS}", "files": chunk, "total_files": total}
        filename = f"chunk_{i+1}.txt"
        with open(filename, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        print(f"💾 {filename}: {len(chunk)} files (start {start+1}-{end}/{total})")

if __name__ == "__main__":
    if not AUDIT_PATH.exists():
        print("⚠️ COMPLETE_FILE_AUDIT.json missing—run audit first.")
    else:
        with open(AUDIT_PATH, 'r') as f:
            data = json.load(f)
        files = flatten_files(data)
        chunk_and_save(files)
        print("✅ 18 chunks ready—paste to DeepSeek sequential.")