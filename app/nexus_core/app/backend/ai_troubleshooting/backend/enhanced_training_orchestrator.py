# enhanced_training_orchestrator.py
import threading
import time
from data_scraper import TroubleshootingDataScraper

class EnhancedTrainingOrchestrator:
    def __init__(self, model_router, experience_evaluator):
        self.model_router = model_router
        self.evaluator = experience_evaluator
        self.data_scraper = TroubleshootingDataScraper()
        self.training_data = []
        self.puppet_stage = "apprentice"  # apprentice → journeyman → viren_ms
        
        # Start background processes
        self._start_background_scraping()
        self._start_continuous_training()
    
    def _start_background_scraping(self):
        """Start data scraping in background"""
        def scraping_loop():
            self.data_scraper.continuous_scraping()
        
        scraping_thread = threading.Thread(target=scraping_loop, daemon=True)
        scraping_thread.start()
        print("🔍 Background data scraping started...")
    
    def _start_continuous_training(self):
        """Start continuous training when system is idle"""
        def training_loop():
            while True:
                if self._should_train():
                    self._incremental_training()
                time.sleep(300)  # Check every 5 minutes
        
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()
        print("🎓 Continuous training monitor started...")
    
    def _should_train(self):
        """Check if we should train (system idle + new data)"""
        if not self._is_system_idle():
            return False
        
        # Check if we have new training data
        dataset_files = list(self.data_scraper.dataset_path.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in dataset_files)
        
        return total_size > 1024 * 1024  # At least 1MB of new data
    
    def _is_system_idle(self):
        """Check if system has 10-20% resources available"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        return cpu_usage < 80 and memory_usage < 80
    
    def _incremental_training(self):
        """Run compactifi training on new data"""
        print("🔄 Running incremental training...")
        
        try:
            # Merge new data into training dataset
            self._merge_training_data()
            
            # Run compactifi training
            subprocess.run([
                "python", "train.py",
                "--dataset", "datasets/viren_training/merged_training.jsonl",
                "--output", f"models/viren_ms_{self.puppet_stage}",
                "--epochs", "1",  # Quick incremental training
                "--quant_level", "int4"  # Fast training
            ], timeout=600)  # 10 minute timeout
            
            print(f"✅ Incremental training complete - Stage: {self.puppet_stage}")
            
            # Advance puppet stage if ready
            self._advance_puppet_stage()
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    def _merge_training_data(self):
        """Merge all scraped data into training format"""
        merged_data = []
        
        # Convert scraped data to training examples
        for category_file in self.data_scraper.dataset_path.glob("*.jsonl"):
            with category_file.open('r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    training_example = self._convert_to_training_example(data)
                    if training_example:
                        merged_data.append(training_example)
        
        # Save merged dataset
        merged_path = self.data_scraper.dataset_path / "merged_training.jsonl"
        with merged_path.open('w', encoding='utf-8') as f:
            for example in merged_data:
                f.write(json.dumps(example) + '\n')
    
    def _convert_to_training_example(self, raw_data):
        """Convert scraped data to training examples"""
        # This would convert system forensics, cloud scenarios, etc.
        # into proper training format for the model
        return {
            "input": f"Troubleshoot: {raw_data.get('scenario', 'system_issue')}",
            "output": self._generate_expected_output(raw_data),
            "metadata": {
                "source": "scraped_data",
                "category": "troubleshooting",
                "difficulty": "intermediate"
            }
        }
    
    def _advance_puppet_stage(self):
        """Advance puppet to next competency stage"""
        stages = ["apprentice", "journeyman", "viren_ms"]
        current_index = stages.index(self.puppet_stage)
        
        if current_index < len(stages) - 1:
            # Check if ready to advance (training data size, performance metrics)
            dataset_size = sum(f.stat().st_size for f in self.data_scraper.dataset_path.glob("*.jsonl"))
            if dataset_size > 50 * 1024 * 1024:  # 50MB of training data
                self.puppet_stage = stages[current_index + 1]
                print(f"🎉 PUPPET ADVANCED TO: {self.puppet_stage.upper()}")