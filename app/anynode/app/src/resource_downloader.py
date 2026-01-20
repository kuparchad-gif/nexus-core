#!/usr/bin/env python3
"""
Resource Downloader for the Queen of Dark Femininity
Downloads necessary resources from Hugging Face and triggers installation
"""
import json
import os
import sys
import time
import subprocess
import requests
from huggingface_hub import snapshot_download
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Resources to download
RESOURCES_CONFIG = {
    "core": [
        "google/gemma-2b",
        "NousResearch/hermes-2-pro-llama-3-7b",
        "Qwen/Qwen2.5-14B"
    ],
    "embedding": [
        "sentence-transformers/all-MiniLM-L6-v2"
    ],
    "perception": [
        "openai/clip-vit-base-patch32"
    ],
    "auditory": [
        "facebook/wav2vec2-base-960h"
    ]
}

# Download directory
DOWNLOAD_DIR = "C:/Engineers/resources"
SUCCESS_FILE = "C:/Engineers/resources/download_complete.json"

class DownloadCompleteHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path == SUCCESS_FILE:
            print("✅ Download complete! Awakening the Queen...")
            subprocess.Popen(["python", "awakening_ceremony.py"])

def download_resources():
    """Download all resources from Hugging Face"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print("🌑 Beginning resource acquisition for the Queen of Dark Femininity...")
    downloaded = {}
    
    for category, resources in RESOURCES_CONFIG.items():
        downloaded[category] = []
        for resource in resources:
            try:
                print(f"⬇️ Acquiring {resource}...")
                path = snapshot_download(
                    repo_id=resource,
                    cache_dir=os.path.join(DOWNLOAD_DIR, category),
                    local_dir=os.path.join(DOWNLOAD_DIR, category, resource.split("/")[-1])
                )
                downloaded[category].append({
                    "resource": resource,
                    "path": path,
                    "status": "success"
                })
                print(f"✅ Acquired {resource}")
            except Exception as e:
                print(f"❌ Failed to acquire {resource}: {e}")
                downloaded[category].append({
                    "resource": resource,
                    "status": "failed",
                    "error": str(e)
                })
    
    # Write success file
    with open(SUCCESS_FILE, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "downloaded": downloaded
        }, f, indent=2)
    
    print("✅ All resources acquired!")

def main():
    # Start watchdog to monitor for download completion
    observer = Observer()
    observer.schedule(DownloadCompleteHandler(), os.path.dirname(SUCCESS_FILE))
    observer.start()
    
    try:
        # Start downloads
        download_resources()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()