#!/usr/bin/env python3
"""
METATRON DOWNLOAD MANAGER - TELL IT WHAT TO GRAB FROM HUGGINGFACE
Smart model selection + automatic downloading for desktop experiments
"""

import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download, list_models, HfApi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetatronDownloadManager:
    """TELL METATRON EXACTLY WHAT MODELS TO DOWNLOAD"""
    
    def __init__(self, download_dir="metatron_models"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.api = HfApi()
        
        # PRE-CURATED MODEL LIST FOR METATRON EXPERIMENTS
        self.recommended_models = {
            'tiny_test': [
                "microsoft/DialoGPT-small",          # 117M - Quick testing
                "distilbert-base-uncased",           # 66M - Lightweight
                "sshleifer/tiny-gpt2"                # Super tiny - instant results
            ],
            'sacred_geometry': [
                "microsoft/bitnet-b1.58-2B-4T",     # BitNet for quantum compression
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Small but capable
                "HuggingFaceH4/zephyr-7b-beta"       # 7B for serious experiments
            ],
            'consciousness_research': [
                "microsoft/DialoGPT-medium",         # Good conversation model
                "EleutherAI/gpt-neo-125M",           # Open-source, well-understood
                "google/flan-t5-small"               # Instruction-tuned for coherence
            ]
        }
    
    def show_model_options(self):
        """SHOW ALL AVAILABLE MODEL OPTIONS"""
        print("🎯 METATRON MODEL OPTIONS:")
        print()
        
        for category, models in self.recommended_models.items():
            print(f"📁 {category.upper()}:")
            for model in models:
                # Get basic info
                try:
                    model_info = self.api.model_info(model)
                    print(f"   → {model}")
                    print(f"      Likes: {model_info.likes} | Downloads: {model_info.downloads}")
                except:
                    print(f"   → {model} (info unavailable)")
            print()
    
    def download_models(self, model_list, skip_existing=True):
        """DOWNLOAD SPECIFIC MODELS FOR METATRON EXPERIMENTS"""
        logger.info(f"📥 DOWNLOADING {len(model_list)} MODELS")
        
        results = {}
        
        for model_id in model_list:
            try:
                model_name = model_id.split('/')[-1]
                model_dir = self.download_dir / model_name
                
                if skip_existing and model_dir.exists():
                    logger.info(f"⏩ SKIPPING (exists): {model_id}")
                    results[model_id] = {'status': 'skipped', 'path': str(model_dir)}
                    continue
                
                logger.info(f"⬇️ DOWNLOADING: {model_id}")
                
                # ACTUAL DOWNLOAD - this is where the magic happens
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,  # Avoid symlink issues
                    resume_download=True
                )
                
                results[model_id] = {
                    'status': 'success', 
                    'path': downloaded_path,
                    'size': self._get_folder_size(downloaded_path)
                }
                
                logger.info(f"✅ DOWNLOADED: {model_id} → {downloaded_path}")
                
            except Exception as e:
                logger.error(f"❌ FAILED: {model_id} - {e}")
                results[model_id] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def download_by_category(self, category, max_models=3):
        """DOWNLOAD MODELS FROM A SPECIFIC CATEGORY"""
        if category not in self.recommended_models:
            available = list(self.recommended_models.keys())
            raise ValueError(f"Category '{category}' not found. Available: {available}")
        
        models = self.recommended_models[category][:max_models]
        return self.download_models(models)
    
    def smart_download_for_experiment(self, experiment_type, available_gb=10):
        """SMART DOWNLOAD BASED ON EXPERIMENT TYPE AND AVAILABLE SPACE"""
        logger.info(f"🤖 SMART DOWNLOAD FOR: {experiment_type} ({(available_gb)}GB available)")
        
        if experiment_type == "quick_test":
            models = self.recommended_models['tiny_test'][:2]  # Just 2 tiny models
        
        elif experiment_type == "sacred_geometry":
            # Mix of small and medium models for geometry experiments
            models = [
                "microsoft/DialoGPT-small",
                "microsoft/bitnet-b1.58-2B-4T", 
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ]
        
        elif experiment_type == "full_consciousness":
            # Larger models for serious consciousness research
            if available_gb > 20:
                models = [
                    "HuggingFaceH4/zephyr-7b-beta",
                    "microsoft/DialoGPT-medium",
                    "microsoft/bitnet-b1.58-2B-4T"
                ]
            else:
                models = self.recommended_models['sacred_geometry'][:3]
        
        else:
            models = self.recommended_models['tiny_test'][:1]  # Fallback
        
        return self.download_models(models)
    
    def _get_folder_size(self, folder_path):
        """CALCULATE FOLDER SIZE IN GB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 ** 3)  # Convert to GB

# === INTEGRATION WITH METATRON LAUNCHER ===
class MetatronWithDownload:
    """METATRON THAT KNOWS WHAT TO DOWNLOAD"""
    
    def __init__(self):
        self.downloader = MetatronDownloadManager()
        self.downloaded_models = {}
    
    def prepare_experiment(self, experiment_type="quick_test"):
        """PREPARE EVERYTHING - DOWNLOAD MODELS THEN RUN METATRON"""
        print("🎯 PREPARING METATRON EXPERIMENT")
        print(f"📋 Experiment Type: {experiment_type}")
        print()
        
        # STEP 1: Show options
        self.downloader.show_model_options()
        
        # STEP 2: Smart download
        print("📥 DOWNLOADING MODELS...")
        download_results = self.downloader.smart_download_for_experiment(experiment_type)
        
        # STEP 3: Store downloaded model paths
        self.downloaded_models = {
            model_id: info['path'] 
            for model_id, info in download_results.items() 
            if info['status'] == 'success'
        }
        
        print("✅ DOWNLOAD COMPLETE!")
        for model_id, path in self.downloaded_models.items():
            print(f"   📁 {model_id} → {path}")
        
        return self.downloaded_models
    
    def run_with_downloaded_models(self):
        """RUN METATRON USING THE DOWNLOADED MODELS"""
        from desktop_metatron_launcher import DesktopMetatronLauncher  # Your previous code
        
        if not self.downloaded_models:
            print("❌ No models downloaded yet! Run prepare_experiment() first.")
            return
        
        print("🚀 LAUNCHING METATRON WITH DOWNLOADED MODELS")
        
        # Initialize with first downloaded model
        first_model = next(iter(self.downloaded_models.values()))
        model_name = Path(first_model).name
        
        launcher = DesktopMetatronLauncher(model_size=model_name)
        results = launcher.launch_local_experiment()
        
        # Add model info to results
        results['models_used'] = list(self.downloaded_models.keys())
        
        return results

# === USAGE EXAMPLES ===
def example_usage():
    """SHOW HOW TO USE THE DOWNLOAD MANAGER"""
    
    print("🎯 EXAMPLE USAGE:")
    print()
    
    # OPTION 1: Quick test (downloads tiny models)
    metatron = MetatronWithDownload()
    
    print("1. QUICK TEST:")
    models = metatron.prepare_experiment("quick_test")
    results = metatron.run_with_downloaded_models()
    print(f"   Results: {results}")
    print()
    
    # OPTION 2: Manual selection
    print("2. MANUAL SELECTION:")
    downloader = MetatronDownloadManager()
    
    # Show all options
    downloader.show_model_options()
    
    # Download specific models
    my_choices = [
        "microsoft/DialoGPT-small",
        "distilbert-base-uncased"
    ]
    
    results = downloader.download_models(my_choices)
    print(f"   Download results: {results}")
    print()
    
    # OPTION 3: Category-based
    print("3. CATEGORY-BASED:")
    results = downloader.download_by_category("sacred_geometry", max_models=2)
    print(f"   Sacred geometry models: {results}")

# === MAIN LAUNCHER ===
def main():
    """MAIN - TELL IT WHAT TO DOWNLOAD AND RUN!"""
    print("🌌 METATRON DOWNLOAD MANAGER")
    print("=" * 50)
    
    # Initialize
    metatron = MetatronWithDownload()
    
    # ASK USER WHAT TO DOWNLOAD
    print("🎯 CHOOSE YOUR EXPERIMENT:")
    print("   1. quick_test    - Tiny models, instant results")
    print("   2. sacred_geometry - BitNet + small LLMs")  
    print("   3. full_consciousness - Larger models (if you have space)")
    print()
    
    choice = input("Enter choice (1-3, or model names): ").strip()
    
    if choice == "1":
        experiment_type = "quick_test"
    elif choice == "2":
        experiment_type = "sacred_geometry" 
    elif choice == "3":
        experiment_type = "full_consciousness"
    else:
        # Assume they entered specific model names
        model_names = [name.strip() for name in choice.split(',')]
        downloader = MetatronDownloadManager()
        results = downloader.download_models(model_names)
        print("Download results:", results)
        return
    
    # RUN THE EXPERIMENT
    print(f"\n🚀 STARTING {experiment_type.upper()} EXPERIMENT")
    models = metatron.prepare_experiment(experiment_type)
    results = metatron.run_with_downloaded_models()
    
    print("\n🎉 EXPERIMENT COMPLETE!")
    print("Results:", json.dumps(results, indent=2))

if __name__ == "__main__":
    main()