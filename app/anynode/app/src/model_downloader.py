#!/usr/bin/env python3
"""
Hugging Face Model Downloader
Downloads models based on JSON config and triggers installation
"""
import json
import os
import sys
import time
import subprocess
import requests
from huggingface_hub import hf_hub_download, snapshot_download
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Models to download
MODELS_CONFIG = {
    "llm": [
        "google/gemma-2b",
        "NousResearch/hermes-2-pro-llama-3-7b",
        "Qwen/Qwen2.5-14B"
    ],
    "embedding": [
        "sentence-transformers/all-MiniLM-L6-v2"
    ],
    "vision": [
        "openai/clip-vit-base-patch32"
    ],
    "audio": [
        "facebook/wav2vec2-base-960h"
    ]
}

# Download directory
DOWNLOAD_DIR = "C:/Engineers/models"
SUCCESS_FILE = "C:/Engineers/models/download_complete.json"

class DownloadCompleteHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path == SUCCESS_FILE:
            print("✅ Download complete! Triggering installation...")
            subprocess.Popen(["python", "model_installer.py"])

def download_models():
    """Download all models from Hugging Face"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print("🚀 Starting model downloads from Hugging Face...")
    downloaded = {}
    
    for category, models in MODELS_CONFIG.items():
        downloaded[category] = []
        for model in models:
            try:
                print(f"⬇️ Downloading {model}...")
                path = snapshot_download(
                    repo_id=model,
                    cache_dir=os.path.join(DOWNLOAD_DIR, category),
                    local_dir=os.path.join(DOWNLOAD_DIR, category, model.split("/")[-1])
                )
                downloaded[category].append({
                    "model": model,
                    "path": path,
                    "status": "success"
                })
                print(f"✅ Downloaded {model}")
            except Exception as e:
                print(f"❌ Failed to download {model}: {e}")
                downloaded[category].append({
                    "model": model,
                    "status": "failed",
                    "error": str(e)
                })
    
    # Write success file
    with open(SUCCESS_FILE, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "downloaded": downloaded
        }, f, indent=2)
    
    print("✅ All downloads complete!")

def main():
    # Start watchdog to monitor for download completion
    observer = Observer()
    observer.schedule(DownloadCompleteHandler(), os.path.dirname(SUCCESS_FILE))
    observer.start()
    
    try:
        # Start downloads
        download_models()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()