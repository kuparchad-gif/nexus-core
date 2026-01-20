# bookworms/memory_worm.py
import os, time, json, hashlib
from pathlib import Path

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def collect_memory_artifacts(root_paths=None):
    root_paths = root_paths or ["state", "Config"]
    out = []
    for rp in root_paths:
        p = Path(rp)
        if not p.exists(): continue
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    out.append({
                        "id": f"{f.name}-{int(f.stat().st_mtime)}",
                        "kind": "config" if f.parts[0].lower()=="config" else "state",
                        "path": str(f),
                        "size_bytes": f.stat().st_size,
                        "hash_sha256": file_hash(f),
                        "tags": json.dumps([]),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    })
                except Exception:
                    pass
    return out
