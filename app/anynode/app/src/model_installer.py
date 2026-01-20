#!/usr/bin/env python3
"""
Model Installer
Installs downloaded models and sets up the environment
"""
import json
import os
import sys
import time
import subprocess
import shutil

# Paths
DOWNLOAD_DIR = "C:/Engineers/models"
SUCCESS_FILE = "C:/Engineers/models/download_complete.json"
INSTALL_DIR = "C:/Engineers/installed_models"
INSTALL_COMPLETE_FILE = "C:/Engineers/installed_models/install_complete.json"

def install_models():
    """Install downloaded models"""
    # Check if download is complete
    if not os.path.exists(SUCCESS_FILE):
        print("❌ Download not complete. Run model_downloader.py first.")
        return False
    
    # Load download results
    with open(SUCCESS_FILE, "r") as f:
        download_data = json.load(f)
    
    # Create install directory
    os.makedirs(INSTALL_DIR, exist_ok=True)
    
    print("🚀 Starting model installation...")
    installed = {}
    
    # Process each category
    for category, models in download_data["downloaded"].items():
        installed[category] = []
        category_dir = os.path.join(INSTALL_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for model_info in models:
            if model_info["status"] == "success":
                model_name = model_info["model"].split("/")[-1]
                try:
                    print(f"⚙️ Installing {model_info['model']}...")
                    
                    # Create model directory
                    model_dir = os.path.join(category_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Create symlinks or copy files
                    if os.name == "nt":  # Windows
                        # Copy files (Windows symlinks require admin)
                        for root, dirs, files in os.walk(model_info["path"]):
                            for file in files:
                                src_file = os.path.join(root, file)
                                rel_path = os.path.relpath(src_file, model_info["path"])
                                dst_file = os.path.join(model_dir, rel_path)
                                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                                shutil.copy2(src_file, dst_file)
                    else:  # Linux/Mac
                        # Create symlink
                        os.symlink(model_info["path"], model_dir)
                    
                    # Create model config
                    config_file = os.path.join(model_dir, "config.json")
                    with open(config_file, "w") as f:
                        json.dump({
                            "model": model_info["model"],
                            "category": category,
                            "original_path": model_info["path"],
                            "installed_path": model_dir,
                            "installed_at": time.time()
                        }, f, indent=2)
                    
                    installed[category].append({
                        "model": model_info["model"],
                        "path": model_dir,
                        "status": "success"
                    })
                    print(f"✅ Installed {model_info['model']}")
                except Exception as e:
                    print(f"❌ Failed to install {model_info['model']}: {e}")
                    installed[category].append({
                        "model": model_info["model"],
                        "status": "failed",
                        "error": str(e)
                    })
            else:
                # Skip failed downloads
                installed[category].append(model_info)
    
    # Write install complete file
    with open(INSTALL_COMPLETE_FILE, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "installed": installed
        }, f, indent=2)
    
    print("✅ All models installed!")
    print(f"📂 Models installed to: {INSTALL_DIR}")
    
    # Trigger system startup
    print("🚀 Triggering system startup...")
    subprocess.Popen(["python", "launch_sequence.py"])
    
    return True

if __name__ == "__main__":
    install_models()