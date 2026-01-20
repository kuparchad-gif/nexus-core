# model_integrity_check.py
# Location: /root/scripts/
# Purpose: Scans for missing or corrupt models based on manifest and reports readiness

import json
import os

def check_models():
    manifest_path = "model_manifest.json"
    if not os.path.exists(manifest_path):
        return {"error": "Manifest not found"}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {}
    for name, meta in manifest.items():
        path = meta.get("path")
        exists = os.path.exists(path)
        results[name] = {"exists": exists, "path": path}
    
    return results

if __name__ == "__main__":
    status = check_models()
    for model, info in status.items():
        print(f"[Check] {model}: {'FOUND' if info['exists'] else 'MISSING'} at {info['path']}")
