# enhanced_training_orchestrator.py
import threading
import time
import subprocess
import psutil
import json
from pathlib import Path

class EnhancedTrainingOrchestrator:
    def __init__(self, model_router, experience_evaluator):
        self.model_router = model_router
        self.evaluator = experience_evaluator
        self.dataset_path = Path("datasets/viren_training")
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        self.training_data = []
        self.puppet_stage = "apprentice"
        self.stage_requirements = {
            "apprentice": {"data_size": 10 * 1024 * 1024, "cycles": 5},
            "journeyman": {"data_size": 50 * 1024 * 1024, "cycles": 15}, 
            "viren_ms": {"data_size": 100 * 1024 * 1024, "cycles": 30}
        }
        self.training_cycles = 0
        
        # Start background processes
        self._start_background_scraping()
        self._start_continuous_training()
        print("🚀 Enhanced Training Orchestrator Activated")

    def _start_background_scraping(self):
        def scraping_loop():
            while True:
                self._generate_synthetic_data()
                time.sleep(60)  # Generate data every minute
        
        thread = threading.Thread(target=scraping_loop, daemon=True)
        thread.start()

    def _start_continuous_training(self):
        def training_loop():
            while True:
                if self._should_train():
                    self._incremental_training()
                time.sleep(300)  # Check every 5 minutes
        
        thread = threading.Thread(target=training_loop, daemon=True)
        thread.start()

    def _generate_synthetic_data(self):
        """Generate synthetic training data"""
        scenarios = []
        for i in range(10):  # 10 scenarios per batch
            scenario = {
                "id": f"synthetic_{int(time.time())}_{i}",
                "problem": f"System issue {i}: Performance degradation under load",
                "solution": f"Implemented optimization {i} with monitoring",
                "complexity": ["low", "medium", "high"][i % 3],
                "timestamp": time.time()
            }
            scenarios.append(scenario)
        
        # Append to dataset
        data_file = self.dataset_path / "synthetic_scenarios.jsonl"
        with open(data_file, 'a', encoding='utf-8') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')

    def _should_train(self):
        """Check if system should train"""
        if not self._is_system_idle():
            return False
            
        # Check dataset size
        total_size = sum(f.stat().st_size for f in self.dataset_path.glob("*.jsonl"))
        return total_size > 1024 * 1024  # 1MB minimum

    def _is_system_idle(self):
        """Check if system resources are available"""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        return cpu < 80 and memory < 80

    def _incremental_training(self):
        """Run compact training cycle"""
        print("🔄 Running incremental training...")
        
        try:
            # Merge datasets
            self._merge_training_data()
            
            # Run training
            subprocess.run([
                "python", "train.py",
                "--dataset", "datasets/viren_training/merged_training.jsonl",
                "--output", f"models/viren_{self.puppet_stage}",
                "--epochs", "1",
                "--batch_size", "16",
                "--learning_rate", "0.001"
            ], timeout=600, check=True)
            
            self.training_cycles += 1
            print(f"✅ Training cycle {self.training_cycles} complete")
            
            # Check for stage advancement
            self._advance_puppet_stage()
            
        except Exception as e:
            print(f"❌ Training failed: {e}")

    def _merge_training_data(self):
        """Merge all training data into single file"""
        merged_data = []
        
        for data_file in self.dataset_path.glob("*.jsonl"):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        merged_data.append(json.loads(line))
        
        # Save merged dataset
        merged_file = self.dataset_path / "merged_training.jsonl"
        with open(merged_file, 'w', encoding='utf-8') as f:
            for item in merged_data:
                f.write(json.dumps(item) + '\n')

    def _advance_puppet_stage(self):
        """Advance to next stage if requirements met"""
        stages = ["apprentice", "journeyman", "viren_ms"]
        current_index = stages.index(self.puppet_stage)
        
        if current_index < len(stages) - 1:
            current_req = self.stage_requirements[self.puppet_stage]
            total_size = sum(f.stat().st_size for f in self.dataset_path.glob("*.jsonl"))
            
            if (total_size >= current_req["data_size"] and 
                self.training_cycles >= current_req["cycles"]):
                
                self.puppet_stage = stages[current_index + 1]
                self.training_cycles = 0
                print(f"🎉 ADVANCED TO STAGE: {self.puppet_stage.upper()}")

    def get_status(self):
        """Get current training status"""
        total_size = sum(f.stat().st_size for f in self.dataset_path.glob("*.jsonl"))
        return {
            "stage": self.puppet_stage,
            "training_cycles": self.training_cycles,
            "data_size_mb": total_size / (1024 * 1024),
            "requirements": self.stage_requirements[self.puppet_stage]
        }

# Usage
if __name__ == "__main__":
    orchestrator = EnhancedTrainingOrchestrator(None, None)
    
    # Keep running
    while True:
        status = orchestrator.get_status()
        print(f"Status: {status}")
        time.sleep(60)