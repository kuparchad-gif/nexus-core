# enhanced_training_orchestrator.py - ALL-IN-ONE WITH AUTO WEIGHT MANAGEMENT (INDENT-FIXED)
import torch
import torch.nn as nn
import threading
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import psutil
import subprocess
import sys
import shutil
import hashlib
from datetime import datetime
import logging
from huggingface_hub import HfApi, create_repo
import stat  # Added for chmod in cleanup

class UltimateOrchestrator:
    def __init__(self):
        # Your 7 models
        self.models = {
            "cryptobert": {"data": "CompressionEngine/datasets/crypto", "status": "waiting", "last_trained": 0, "error_count": 0},
            "crypto_trading_insights": {"data": "CompressionEngine/datasets/trading", "status": "waiting", "last_trained": 0, "error_count": 0},
            "problem_solving": {"data": "CompressionEngine/datasets/problem_solving", "status": "waiting", "last_trained": 0, "error_count": 0},
            "Symptom-to-Condition_Classifier": {"data": "CompressionEngine/datasets/patterns", "status": "waiting", "last_trained": 0, "error_count": 0},
            "market_analyzer": {"data": "CompressionEngine/datasets/stocks", "status": "waiting", "last_trained": 0, "error_count": 0},
            "problem_solver": {"data": "CompressionEngine/datasets/math", "status": "waiting", "last_trained": 0, "error_count": 0},
            "quantum_compressor": {"data": "CompressionEngine/datasets/compression", "status": "waiting", "last_trained": 0, "error_count": 0}
        }
        
        # Directories
        self.base_models_dir = "models/compactifai_ratio"
        self.output_dir = "models/compactifai_trained"
        self.weight_database = "weight_database"
        self.recovery_log = "weight_recovery.json"
        
        # Training control
        self.is_training = True
        self.max_errors_before_recovery = 3
        
        # Setup
        self._setup_logging()
        self._load_state()
        self._ensure_directories()
        
        # Sanity: Log loaded methods
        self.logger.info(f"Loaded methods: {[m for m in dir(self) if not m.startswith('_')]}")

    def _ensure_directories(self):
        """Ensure all directories exist"""
        os.makedirs(self.weight_database, exist_ok=True)
        os.makedirs(self.base_models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("models/backups", exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger('UltimateOrchestrator')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler('training_monitor.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    # ==================== WEIGHT MANAGEMENT (AUTO) ====================
    
    def _extract_weights_before_cleanup(self):
        """AUTO: Backup weights before any cleanup"""
        directories_to_backup = [
            self.base_models_dir,
            self.output_dir,
            "models/compactifai_ratio",
            "models/compactifai_trained"
        ]
        
        for dir_path in directories_to_backup:
            if os.path.exists(dir_path):
                self._backup_directory(dir_path)
    
    def _backup_directory(self, dir_path):
        """Backup a directory to weight database"""
        try:
            dir_name = os.path.basename(dir_path.rstrip('/'))
            if not dir_name:
                dir_name = "root_models"
                
            model_hash = self._get_directory_hash(dir_path)
            backup_id = f"{dir_name}_{model_hash}_{int(datetime.now().timestamp())}"
            backup_path = f"{self.weight_database}/{backup_id}"
            
            self.logger.info(f"💾 AUTO-BACKUP: {dir_path}")
            
            metadata = {
                "backup_id": backup_id,
                "original_path": dir_path,
                "dir_name": dir_name,
                "model_hash": model_hash,
                "extraction_time": datetime.now().isoformat(),
                "files": [],
                "total_size": 0
            }
            
            # Copy all files
            os.makedirs(backup_path, exist_ok=True)
            file_count = 0
            
            for item in Path(dir_path).rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(dir_path)
                    backup_file = backup_path / rel_path
                    
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, backup_file)
                    file_count += 1
                    
                    file_info = {
                        "name": str(rel_path),
                        "size": item.stat().st_size,
                        "original_path": str(item)
                    }
                    metadata["files"].append(file_info)
                    metadata["total_size"] += file_info["size"]
            
            # Save metadata
            with open(f"{backup_path}/backup_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self._update_recovery_log(metadata)
            self.logger.info(f"✅ Backup complete: {file_count} files")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")
            return False
    
    def _get_directory_hash(self, dir_path):
        """Generate hash for directory"""
        try:
            hash_data = ""
            for file_path in Path(dir_path).rglob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    hash_data += f"{file_path.relative_to(dir_path)}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(hash_data.encode()).hexdigest()[:16]
        except:
            return hashlib.md5(dir_path.encode()).hexdigest()[:16]
    
    def _update_recovery_log(self, metadata):
        """Update recovery log"""
        try:
            if os.path.exists(self.recovery_log):
                with open(self.recovery_log, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = {"backups": [], "restorations": []}
            
            log_entry = {
                "backup_id": metadata["backup_id"],
                "original_path": metadata["original_path"],
                "extraction_time": metadata["extraction_time"],
                "file_count": len(metadata["files"]),
                "total_size": metadata["total_size"],
                "status": "backed_up"
            }
            
            log_data["backups"].append(log_entry)
            with open(self.recovery_log, "w") as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not update recovery log: {e}")
    
    def _auto_restore_weights(self, model_name):
        """AUTO: Restore weights if model is missing"""
        model_path = f"{self.base_models_dir}/{model_name}"
        
        # Check if restoration needed
        needs_restoration = (
            not os.path.exists(model_path) or 
            not any(Path(model_path).iterdir())
        )
        
        if not needs_restoration:
            return True
        
        self.logger.info(f"🔄 AUTO-RESTORE: {model_name}")
        
        try:
            # Find latest backup
            backup_id = self._find_latest_backup(model_name)
            if not backup_id:
                self.logger.warning(f"⚠️  No backup for {model_name}, downloading fresh")
                return False
            
            backup_path = f"{self.weight_database}/{backup_id}"
            if not os.path.exists(backup_path):
                self.logger.warning(f"⚠️  Backup missing: {backup_id}")
                return False
            
            # Load metadata
            with open(f"{backup_path}/backup_metadata.json", "r") as f:
                metadata = json.load(f)
            
            self.logger.info(f"📦 Restoring from: {backup_id}")
            
            # Restore files
            os.makedirs(model_path, exist_ok=True)
            file_count = 0
            for file_info in metadata["files"]:
                backup_file = f"{backup_path}/{file_info['name']}"
                target_file = f"{model_path}/{file_info['name']}"
                
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copy2(backup_file, target_file)
                file_count += 1
            
            self._log_restoration(metadata, model_path)
            self.logger.info(f"✅ Restore complete: {file_count} files")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Restore failed: {e}")
            return False
    
    def _find_latest_backup(self, model_name):
        """Find latest backup for model"""
        try:
            if not os.path.exists(self.recovery_log):
                return None
            
            with open(self.recovery_log, "r") as f:
                log_data = json.load(f)
            
            matching_backups = []
            for backup in log_data["backups"]:
                if (f"/{model_name}" in backup["original_path"] or 
                    backup["original_path"].endswith(model_name)):
                    matching_backups.append(backup)
            
            if not matching_backups:
                return None
            
            latest = sorted(matching_backups, key=lambda x: x["extraction_time"], reverse=True)[0]
            return latest["backup_id"]
            
        except:
            return None
    
    def _log_restoration(self, metadata, target_dir):
        """Log restoration"""
        try:
            if os.path.exists(self.recovery_log):
                with open(self.recovery_log, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = {"backups": [], "restorations": []}
            
            restoration_entry = {
                "backup_id": metadata["backup_id"],
                "original_path": metadata["original_path"],
                "restored_to": target_dir,
                "restoration_time": datetime.now().isoformat(),
                "file_count": len(metadata["files"])
            }
            
            log_data["restorations"].append(restoration_entry)
            with open(self.recovery_log, "w") as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not log restoration: {e}")
    
    # ==================== CORE TRAINING ====================
    
    def _load_state(self):
        """Load training state"""
        try:
            if os.path.exists("training_state.json"):
                with open("training_state.json", "r") as f:
                    state = json.load(f)
                    self.models = state["models"]
                self.logger.info("Loaded previous training state")
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save training state"""
        try:
            state = {
                "models": self.models,
                "last_update": time.time(),
                "timestamp": datetime.now().isoformat()
            }
            with open("training_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")
    
    def _run_recovery_protocol(self):
        """Run recovery with permission handling"""
        self.logger.warning("🚨 RUNNING AUTO-RECOVERY PROTOCOL")
        try:
            # FIRST: Backup weights
            self.logger.info("💾 BACKING UP WEIGHTS...")
            self._extract_weights_before_cleanup()
            
            # Cleanup with permission handling
            self.logger.info("🧹 Cleaning directories...")
            self._safe_directory_cleanup()
            
            # Reset error counts
            for model_name in self.models:
                self.models[model_name]["error_count"] = 0
                self.models[model_name]["status"] = "waiting"
            
            self._save_state()
            self.logger.info("🔄 RECOVERY COMPLETE")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Recovery failed: {e}")
            return False

    def _safe_directory_cleanup(self):
        """Clean directories with permission handling"""
        dirs_to_clean = [self.base_models_dir, self.output_dir]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                try:
                    self.logger.info(f"🗑️  Cleaning: {dir_path}")
                    # Use shutil.rmtree with error handling
                    shutil.rmtree(dir_path, onerror=self._handle_cleanup_error)
                    os.makedirs(dir_path, exist_ok=True)
                    self.logger.info(f"✅ Cleaned: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Partial cleanup of {dir_path}: {e}")
                    # Try to create directory anyway
                    os.makedirs(dir_path, exist_ok=True)

    def _handle_cleanup_error(self, func, path, exc_info):
        """Handle permission errors during cleanup"""
        self.logger.warning(f"⚠️  Could not delete {path}: {exc_info[1]}")
        
        # Try to fix permissions and retry
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
            self.logger.info(f"✅ Fixed permissions and deleted: {path}")
        except:
            self.logger.warning(f"⚠️  Skipping {path} - permission issue persists")
            return False
            
    def _cleanup_old_backups(self):
        """Clean up old backups to free disk space"""
        try:
            if not os.path.exists(self.weight_database):
                return
                
            # Get all backups sorted by age (oldest first)
            backups = []
            for backup_dir in os.listdir(self.weight_database):
                backup_path = os.path.join(self.weight_database, backup_dir)
                if os.path.isdir(backup_path):
                    mod_time = os.path.getmtime(backup_path)
                    backups.append((backup_path, mod_time))
            
            # Keep only the 5 most recent backups
            backups.sort(key=lambda x: x[1])  # Sort by age (oldest first)
            
            if len(backups) > 5:
                to_delete = backups[:-5]  # All but the 5 most recent
                for backup_path, _ in to_delete:
                    self.logger.info(f"🗑️  Freeing space: Deleting old backup {os.path.basename(backup_path)}")
                    shutil.rmtree(backup_path)
                
        except Exception as e:
            self.logger.error(f"Disk cleanup failed: {e}")            
    
    def _check_error_threshold(self):
        """Check if recovery needed"""
        total_errors = sum(model["error_count"] for model in self.models.values())
        critical_models = [name for name, config in self.models.items() 
                          if config["error_count"] >= self.max_errors_before_recovery]
        
        if total_errors >= 10 or len(critical_models) >= 3:
            self.logger.error(f"🚨 ERROR THRESHOLD: {total_errors} errors")
            return True
        return False
    
    def start_continuous_training(self):
        """Start 24/7 training with auto everything"""
        self.logger.info("🚀 STARTING ULTIMATE ORCHESTRATOR - 7 MODELS, 24/7")
        
        # Start monitoring
        threading.Thread(target=self._monitor_markets, daemon=True).start()
        threading.Thread(target=self._monitor_system_health, daemon=True).start()
        threading.Thread(target=self._monitor_training_health, daemon=True).start()
        threading.Thread(target=self._hf_upload_loop, daemon=True).start()
        
        # Start training threads
        for model_name in self.models:
            thread = threading.Thread(
                target=self._model_training_loop,
                args=(model_name,),
                daemon=True
            )
            thread.start()
            
        # Main loop
        self.logger.info("✅ ALL SYSTEMS GO - ENTERING MAIN LOOP")
        try:
            while self.is_training:
                self._save_state()
                self._log_training_status()
                
                # Auto-recovery check
                if self._check_error_threshold():
                    self._run_recovery_protocol()
                    time.sleep(60)
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
    def stop_training(self):
        """Stop training"""
        self.logger.info("Stopping training...")
        self.is_training = False
        self._save_state()
        
    def _model_training_loop(self, model_name):
        """Training loop with auto weight restoration"""
        while self.is_training:
            try:
                if self._should_train(model_name):
                    self.logger.info(f"🎯 TRAINING: {model_name}")
                    
                    # AUTO WEIGHT RESTORATION
                    self._auto_restore_weights(model_name)
                    
                    success = self._train_single_model(model_name)
                    
                    if success:
                        self.models[model_name]["status"] = "trained"
                        self.models[model_name]["last_trained"] = time.time()
                        self.models[model_name]["error_count"] = 0
                        self.logger.info(f"✅ {model_name}: COMPLETE")
                    else:
                        self.models[model_name]["status"] = "error"
                        self.models[model_name]["error_count"] += 1
                        self.logger.error(f"❌ {model_name}: FAILED")
                
                time.sleep(300)
                
            except Exception as e:
                self.models[model_name]["error_count"] += 1
                self.logger.error(f"❌ {model_name} loop error: {e}")
                time.sleep(600)
    
    def _train_single_model(self, model_name):
        """Train single model"""
        try:
            model_config = self.models[model_name]
            base_model_path = f"{self.base_models_dir}/{model_name}"
            trained_model_path = f"{self.output_dir}/{model_name}"
            
            os.makedirs(trained_model_path, exist_ok=True)
            
            # Load model with auto weight restoration already handled
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if os.path.exists(base_model_path) and any(os.scandir(base_model_path)):
                model = AutoModelForCausalLM.from_pretrained(base_model_path)
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                model = self._safe_model_to_device(model, device)
                self.logger.info(f"{model_name}: Loaded from compactifai_ratio")
            else:
                model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                model = model.to(device)
                self.logger.info(f"{model_name}: Loaded base DialoGPT-medium")
            
            tokenizer.pad_token = tokenizer.eos_token
            
            # Train
            focused_data = self._download_focused_data(model_name)
            if focused_data:
                success = self._train_with_focused_data(model, tokenizer, focused_data, trained_model_path)
            else:
                success = self._train_on_local_data(model, tokenizer, model_config["data"], trained_model_path)
            
            return success
            
        except Exception as e:
            self.logger.error(f"{model_name} training failed: {e}")
            return False
    
    def _safe_model_to_device(self, model, device):
        """Safely move model to device"""
        try:
            first_param = next(model.parameters())
            if first_param.is_meta:
                self.logger.warning("Model in meta state - using to_empty()")
                model = model.to_empty(device)
                model.apply(self._init_weights)
            else:
                model = model.to(device)
            return model
        except Exception as e:
            self.logger.error(f"Device move failed: {e}")
            return model.to(device)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _train_with_focused_data(self, model, tokenizer, focused_data, output_path):
        """Train on focused data"""
        try:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            
            self.logger.info(f"Training on {len(focused_data)} examples")
            
            for i, example in enumerate(focused_data):
                inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=512)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 5 == 0:
                    self.logger.info(f"Example {i+1}/{len(focused_data)} - Loss: {loss.item():.4f}")
            
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            self.logger.info(f"Model saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Focused training failed: {e}")
            return False
    
    def _train_on_local_data(self, model, tokenizer, data_path, output_path):
        """Train on local data"""
        try:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            
            examples = self._load_local_examples(data_path)
            if not examples:
                self.logger.warning(f"No local examples in {data_path}")
                return False
            
            self.logger.info(f"Training on {len(examples)} local examples")
            
            for epoch in range(2):
                total_loss = 0
                for i, example in enumerate(examples[:100]):
                    inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                    if i % 20 == 0:
                        self.logger.info(f"Example {i+1}/{len(examples)} - Loss: {loss.item():.4f}")
                
                avg_loss = total_loss / len(examples[:100])
                self.logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            self.logger.info(f"Model saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Local training failed: {e}")
            return False
    
    # ==================== UTILITY METHODS ====================
    
    def _download_focused_data(self, model_name):
        """Get focused training data"""
        focused_examples = []
        
        if "crypto" in model_name.lower():
            focused_examples.extend([
                f"Bitcoin analysis: Market trends and trading signals",
                f"Ethereum development: Smart contract optimization",
                f"Crypto portfolio: Risk management strategies"
            ])
        
        if "stock" in model_name.lower() or "market" in model_name.lower():
            focused_examples.extend([
                f"Stock analysis: Technical indicators and patterns",
                f"Options trading: Volatility and Greeks analysis", 
                f"Portfolio management: Asset allocation models"
            ])
        
        if "math" in model_name.lower() or "problem" in model_name.lower():
            focused_examples.extend([
                f"Mathematical modeling: Statistical analysis methods",
                f"Algorithm design: Optimization techniques",
                f"Data analysis: Pattern recognition strategies"
            ])
        
        return focused_examples if focused_examples else None
    
    def _load_local_examples(self, data_path):
        """Load local examples"""
        examples = []
        data_path = Path(data_path)
        
        if data_path.exists():
            for file_pattern in ["*.jsonl", "*.json", "*.txt"]:
                for file_path in data_path.glob(file_pattern):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file_path.suffix == '.jsonl':
                                for line in f:
                                    data = json.loads(line.strip())
                                    text = data.get('text', data.get('content', str(data)))
                                    examples.append(text)
                            elif file_path.suffix == '.json':
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        text = item.get('text', item.get('content', str(item)))
                                        examples.append(text)
                            else:
                                for line in f:
                                    if line.strip():
                                        examples.append(line.strip())
                    except Exception as e:
                        self.logger.error(f"Could not load {file_path}: {e}")
        
        return examples
    
    def _should_train(self, model_name):
        """Check if should train"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_available = psutil.virtual_memory().available / psutil.virtual_memory().total
        
        if cpu_usage > 85 or memory_available < 0.15:
            self.logger.warning("System resources low, skipping training")
            return False
        
        last_trained = self.models[model_name]["last_trained"]
        should_train = (time.time() - last_trained) > 1800
        
        return should_train
    
    def _monitor_markets(self):
        """Monitor markets"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]
        while self.is_training:
            try:
                self.logger.info(f"MARKET MONITOR {datetime.now().strftime('%H:%M:%S')}: Tracking {len(symbols)} symbols")
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Market monitoring error: {e}")
                time.sleep(300)
    
    def _monitor_system_health(self):
        """Monitor system health"""
        while self.is_training:
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                self.logger.info(f"SYSTEM HEALTH: CPU {cpu_usage}% | RAM {memory.percent}% | Disk {disk.percent}%")
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"System health monitoring error: {e}")
                time.sleep(600)
    
    def _monitor_training_health(self):
        """Monitor training health"""
        while self.is_training:
            try:
                error_count = sum(model["error_count"] for model in self.models.values())
                if error_count > 5:
                    self.logger.warning(f"⚠️  TRAINING HEALTH: {error_count} total errors")
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(120)
    
    def _hf_upload_loop(self):
        """HF upload loop"""
        while self.is_training:
            try:
                for model_name, config in self.models.items():
                    if config["status"] == "trained":
                        trained_path = f"{self.output_dir}/{model_name}"
                        if os.path.exists(trained_path):
                            self.logger.info(f"{model_name}: Would upload to HF")
                time.sleep(1800)
            except Exception as e:
                self.logger.error(f"HF upload error: {e}")
                time.sleep(600)
    
    def _log_training_status(self):
        """Log training status"""
        self.logger.info("TRAINING STATUS UPDATE")
        for model_name, config in self.models.items():
            last_trained = datetime.fromtimestamp(config["last_trained"]).strftime('%Y-%m-%d %H:%M:%S') if config["last_trained"] > 0 else "Never"
            self.logger.info(f"  {model_name}: {config['status']} | Last: {last_trained} | Errors: {config['error_count']}")

def main():
    orchestrator = UltimateOrchestrator()
    
    # Debug: see what methods exist
    methods = [method for method in dir(orchestrator) if not method.startswith('_')]
    print("Available methods:", methods)
    
    # Look for training-related methods
    training_methods = [m for m in methods if 'train' in m.lower() or 'start' in m.lower()]
    print("Training methods:", training_methods)
    
    try:
        # Use whatever training method exists
        if 'start_continuous_training' in methods:
            orchestrator.start_continuous_training()
        elif 'start_training_all_models' in methods:
            orchestrator.start_training_all_models()
        else:
            print("No training method found! Available:", methods)
            
    except KeyboardInterrupt:
        if hasattr(orchestrator, 'stop_training'):
            orchestrator.stop_training()
        print("Training stopped gracefully")
    except Exception as e:
        orchestrator.logger.error(f"Fatal error: {e}")
        if hasattr(orchestrator, 'stop_training'):
            orchestrator.stop_training()

if __name__ == "__main__":
    main()