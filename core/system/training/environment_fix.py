# environment_fix.py
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def clean_environment():
    print("🧹 CLEANING TRANSFORMERS CACHE AND FIXING ENVIRONMENT")
    
    # Clear cache directories
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        "models/compactifai_ratio"  # Your problematic directory
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                print(f"Cleaning: {cache_dir}")
                shutil.rmtree(cache_dir)
            except Exception as e:
                print(f"Could not clean {cache_dir}: {e}")
    
    print("✅ Environment cleaned")
    
    # Test fresh model load
    print("🧪 Testing fresh model load...")
    try:
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        print("✅ Fresh model load SUCCESSFUL!")
        return True
    except Exception as e:
        print(f"❌ Fresh model load failed: {e}")
        return False

if __name__ == "__main__":
    clean_environment()