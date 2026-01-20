# C:\Engineers\eden_engineering\models\scan_models.py
# Scans actual filesystem for .gguf files

import os
import json
from pathlib import Path

def scan_actual_models(models_dir="C:/Engineers/eden_engineering/models"):
    """Scan the actual filesystem for .gguf files"""
    found_models = []
    
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.gguf'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, models_dir)
                
                # Parse path structure: publisher/model-GGUF/filename.gguf
                path_parts = rel_path.split(os.sep)
                if len(path_parts) >= 3:
                    publisher = path_parts[0]
                    model_dir = path_parts[1]
                    filename = path_parts[2]
                    
                    found_models.append({
                        "full_path": rel_path.replace(os.sep, "/"),
                        "publisher": publisher,
                        "model_dir": model_dir,
                        "filename": filename,
                        "file_size_gb": round(os.path.getsize(full_path) / (1024**3), 2)
                    })
    
    return found_models

def compare_manifest_vs_actual():
    """Compare what's in manifest vs what's actually downloaded"""
    
    # Get manifest models
    try:
        with open("C:/Engineers/eden_engineering/models/model_manifest.json", "r") as f:
            manifest_data = json.load(f)
            manifest_models = manifest_data.get("data", [])
    except:
        manifest_models = []
    
    # Get actual files
    actual_models = scan_actual_models()
    
    print("📁 ACTUAL FILES FOUND:")
    for model in actual_models:
        print(f"  {model['full_path']} ({model['file_size_gb']}GB)")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Manifest entries: {len(manifest_models)}")
    print(f"  Actual files: {len(actual_models)}")
    
    return actual_models

if __name__ == "__main__":
    compare_manifest_vs_actual()