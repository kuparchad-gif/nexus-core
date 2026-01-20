# data_hunter.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import glob

class DataHunter:
    """I don't care where the data is - I find it and use it"""
    
    def __init__(self):
        self.log = logging.getLogger("DataHunter")
        self.found_datasets = {}
        self.training_sources = []
        
    def hunt_all_data(self, root_path: str = None) -> Dict:
        """Hunt for ALL training data regardless of location"""
        if not root_path:
            # Hunt everywhere sensible
            root_path = self._get_sensible_roots()
        
        print("🔍 DATA HUNTER ACTIVATED - I DON'T CARE WHERE IT IS")
        
        # Hunt in parallel across all locations
        hunt_results = {}
        for location in root_path:
            hunt_results[location] = self._hunt_in_location(location)
        
        # Consolidate findings
        self._consolidate_findings(hunt_results)
        
        print(f"🎯 HUNT COMPLETE: Found {len(self.training_sources)} data sources")
        return self.found_datasets
    
    def _get_sensible_roots(self) -> List[str]:
        """Get sensible places to hunt for data"""
        roots = []
        
        # Current directory and parents
        current = Path.cwd()
        for i in range(3):  # Check current + 2 parents up
            roots.append(str(current))
            current = current.parent
        
        # Common data locations
        common_spots = [
            "~/Desktop", "~/Documents", "~/Downloads",
            "/data", "/datasets", "/tmp",
            "C:/Users/Admin/Desktop", "C:/Users/Admin/Documents",
            "D:/", "E:/"
        ]
        
        for spot in common_spots:
            expanded = Path(spot).expanduser()
            if expanded.exists():
                roots.append(str(expanded))
        
        # Also hunt for any folder containing "data", "dataset", "train", "model"
        wildcard_hunts = [
            "**/data*/**/*.jsonl",
            "**/dataset*/**/*.json", 
            "**/train*/**/*.jsonl",
            "**/model*/**/*.json",
            "**/*.jsonl",
            "**/*training*/*.json"
        ]
        
        return list(set(roots))  # Remove duplicates
    
    def _hunt_in_location(self, location: str) -> Dict:
        """Hunt for data in a specific location"""
        print(f"  🕵️ Hunting in: {location}")
        findings = {}
        
        try:
            path = Path(location)
            if not path.exists():
                return findings
            
            # Look for all JSON/JSONL files recursively
            json_patterns = ["**/*.jsonl", "**/*.json", "**/*.txt", "**/*.csv"]
            
            for pattern in json_patterns:
                for file_path in path.glob(pattern):
                    if self._looks_like_training_data(file_path):
                        findings[str(file_path)] = {
                            "size": file_path.stat().st_size,
                            "type": self._classify_data_type(file_path),
                            "samples": self._count_samples(file_path),
                            "usable": True
                        }
                        print(f"    ✅ Found: {file_path} ({findings[str(file_path)]['samples']} samples)")
            
            # Also look for directories that might contain datasets
            for dir_path in path.iterdir():
                if dir_path.is_dir() and self._looks_like_dataset_dir(dir_path):
                    dir_findings = self._hunt_in_location(str(dir_path))
                    findings.update(dir_findings)
                    
        except Exception as e:
            print(f"    ❌ Hunt failed in {location}: {e}")
        
        return findings
    
    def _looks_like_training_data(self, file_path: Path) -> bool:
        """Check if file looks like training data"""
        try:
            if file_path.suffix in ['.jsonl', '.json']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        return False
                    
                    # Check if it's structured data
                    data = json.loads(first_line)
                    
                    # Common training data patterns
                    if isinstance(data, dict):
                        keys = data.keys()
                        # Look for common training data keys
                        training_indicators = ['input', 'output', 'text', 'label', 'prompt', 'completion']
                        return any(indicator in str(keys).lower() for indicator in training_indicators)
            
            return False
            
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        except Exception:
            return False
    
    def _classify_data_type(self, file_path: Path) -> str:
        """Classify what type of data this is"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                data = json.loads(first_line)
                
                if 'input' in data and 'output' in data:
                    return "instruction_tuning"
                elif 'text' in data:
                    return "text_completion" 
                elif 'prompt' in data and 'completion' in data:
                    return "prompt_completion"
                elif 'question' in data and 'answer' in data:
                    return "qa_pairs"
                else:
                    return "generic_training"
                    
        except:
            return "unknown"
    
    def _count_samples(self, file_path: Path) -> int:
        """Count samples in data file"""
        try:
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        except:
            return 0
    
    def _looks_like_dataset_dir(self, dir_path: Path) -> bool:
        """Check if directory looks like it contains datasets"""
        dir_name = dir_path.name.lower()
        dataset_indicators = ['data', 'dataset', 'train', 'training', 'model', 'ai', 'ml']
        return any(indicator in dir_name for indicator in dataset_indicators)
    
    def _consolidate_findings(self, hunt_results: Dict):
        """Consolidate all hunt results"""
        for location, findings in hunt_results.items():
            for file_path, info in findings.items():
                if info['usable'] and info['samples'] > 0:
                    self.found_datasets[file_path] = info
                    self.training_sources.append(file_path)
    
    def get_training_corpus(self) -> List[Dict]:
        """Get all found data as training corpus"""
        corpus = []
        
        for file_path in self.training_sources:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            corpus.append(data)
                            # Stop if we have enough
                            if len(corpus) >= 10000:  # Reasonable limit
                                return corpus
            except Exception as e:
                print(f"⚠️ Couldn't read {file_path}: {e}")
        
        return corpus

# ===== I DON'T CARE TRAINING ORCHESTRATOR =====

class ApatheticTrainingOrchestrator:
    """Trains on whatever data it finds - doesn't care about paths"""
    
    def __init__(self):
        self.hunter = DataHunter()
        self.training_cycles = 0
        self.stage = "apprentice"
        
        print("🤖 APATHETIC TRAINING ORCHESTRATOR INITIALIZED")
        print("   I don't care where the data lives. I'll find it and use it.")
    
    def hunt_and_train(self):
        """Find whatever data exists and train on it"""
        print("\n🎯 STARTING APATHETIC TRAINING CYCLE")
        
        # Hunt for data
        found_data = self.hunter.hunt_all_data()
        
        if not found_data:
            print("😴 No data found. Taking a nap.")
            return {"status": "no_data", "action": "napping"}
        
        # Use whatever we found
        training_corpus = self.hunter.get_training_corpus()
        
        print(f"🔥 Training on {len(training_corpus)} samples from {len(found_data)} sources")
        
        # Train with whatever we have
        result = self._train_on_whatever(training_corpus)
        
        self.training_cycles += 1
        self._maybe_advance_stage()
        
        return {
            "status": "trained",
            "cycles": self.training_cycles,
            "stage": self.stage,
            "data_sources": len(found_data),
            "training_samples": len(training_corpus),
            "sources": list(found_data.keys())[:5]  # First 5 sources
        }
    
    def _train_on_whatever(self, corpus: List[Dict]):
        """Train on whatever data we found"""
        # Your actual training logic here
        # This would call your train.py or whatever
        
        print(f"⚡ Training on {len(corpus)} samples...")
        
        # Simulate training
        import time
        time.sleep(2)  # Simulate training time
        
        # Just use the data somehow
        trained_patterns = set()
        for item in corpus[:1000]:  # Use first 1000 samples
            if isinstance(item, dict):
                trained_patterns.update(str(item).split()[:10])
        
        return {
            "patterns_learned": len(trained_patterns),
            "training_time": "simulated",
            "message": "Trained on whatever data was available"
        }
    
    def _maybe_advance_stage(self):
        """Advance stage based on training cycles"""
        stages = {
            5: "apprentice",
            15: "journeyman", 
            30: "viren_ms"
        }
        
        for threshold, stage in stages.items():
            if self.training_cycles >= threshold and self.stage != stage:
                self.stage = stage
                print(f"🎉 STAGE ADVANCEMENT: {stage.upper()}")
                break

# ===== USAGE =====
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DATA HUNTER + APATHETIC TRAINER")
    print("   I don't care where your data lives")
    print("=" * 60)
    
    # Let the hunter find everything
    hunter = DataHunter()
    found = hunter.hunt_all_data()
    
    print(f"\n📊 FOUND {len(found)} DATA SOURCES:")
    for path, info in list(found.items())[:10]:  # Show first 10
        print(f"   📁 {path}")
        print(f"      samples: {info['samples']}, type: {info['type']}")
    
    # Train on whatever we found
    trainer = ApatheticTrainingOrchestrator()
    result = trainer.hunt_and_train()
    
    print(f"\n🎯 TRAINING RESULT:")
    print(f"   {result}")