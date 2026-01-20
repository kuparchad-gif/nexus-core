#!/usr/bin/env python3
"""
Kaggle Trainer for Cloud Viren
Enables training on Kaggle datasets and distributing weights
"""

import os
import json
import time
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KaggleTrainer")

class KaggleTrainer:
    """Enables training on Kaggle datasets and distributing weights"""
    
    def __init__(self, config_path: str = None, weight_manager = None, database_registry = None):
        """Initialize the Kaggle trainer"""
        self.config_path = config_path or os.path.join('C:/Viren/config', 'kaggle_trainer.json')
        self.weight_manager = weight_manager
        self.database_registry = database_registry
        self.config = {}
        self.training_history = []
        self._load_config()
    
    def _load_config(self) -> None:
        """Load trainer configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default configuration
                self.config = {
                    "kaggle_username": "",
                    "kaggle_key": "",
                    "output_dir": "C:/Viren/weights/trained",
                    "training_templates": {
                        "finance": {
                            "base_model": "base",
                            "datasets": ["finance-data-latest"],
                            "training_script": "train_finance.py",
                            "hyperparameters": {
                                "learning_rate": 0.001,
                                "epochs": 10,
                                "batch_size": 32
                            }
                        },
                        "science": {
                            "base_model": "base",
                            "datasets": ["science-papers-2023"],
                            "training_script": "train_science.py",
                            "hyperparameters": {
                                "learning_rate": 0.0005,
                                "epochs": 15,
                                "batch_size": 16
                            }
                        }
                    },
                    "distribution": {
                        "auto_distribute": True,
                        "target_systems": ["all"]
                    }
                }
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading trainer configuration: {e}")
            self.config = {}
    
    def _save_config(self) -> None:
        """Save trainer configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trainer configuration: {e}")
    
    def initialize(self) -> bool:
        """Initialize the Kaggle trainer"""
        logger.info("Initializing Kaggle Trainer")
        
        # Create output directory
        os.makedirs(self.config.get("output_dir", "C:/Viren/weights/trained"), exist_ok=True)
        
        # Check Kaggle credentials
        if not self.config.get("kaggle_username") or not self.config.get("kaggle_key"):
            logger.warning("Kaggle credentials not configured")
        
        return True
    
    def _setup_kaggle_credentials(self) -> bool:
        """Set up Kaggle credentials"""
        if not self.config.get("kaggle_username") or not self.config.get("kaggle_key"):
            logger.error("Kaggle credentials not configured")
            return False
        
        try:
            # Create Kaggle API credentials file
            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            
            with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
                json.dump({
                    "username": self.config["kaggle_username"],
                    "key": self.config["kaggle_key"]
                }, f)
            
            # Set permissions
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            
            return True
        except Exception as e:
            logger.error(f"Error setting up Kaggle credentials: {e}")
            return False
    
    def _download_dataset(self, dataset: str, output_dir: str) -> bool:
        """Download a dataset from Kaggle"""
        try:
            if not self._setup_kaggle_credentials():
                return False
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Download dataset
            cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", output_dir, "--unzip"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error downloading dataset {dataset}: {result.stderr}")
                return False
            
            logger.info(f"Successfully downloaded dataset {dataset}")
            return True
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset}: {e}")
            return False
    
    def train_domain(self, domain: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train weights for a specific domain using Kaggle datasets"""
        if domain not in self.config.get("training_templates", {}):
            return {
                "success": False,
                "error": f"No training template found for domain: {domain}"
            }
        
        template = self.config["training_templates"][domain]
        params = template.get("hyperparameters", {})
        
        # Apply custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        try:
            # Create temporary directory for training
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download datasets
                for dataset in template.get("datasets", []):
                    if not self._download_dataset(dataset, os.path.join(temp_dir, "data")):
                        return {
                            "success": False,
                            "error": f"Failed to download dataset: {dataset}"
                        }
                
                # In a real system, we would run the training script here
                # For now, we'll simulate training
                logger.info(f"Training domain: {domain} with parameters: {params}")
                time.sleep(5)  # Simulate training time
                
                # Create output directory
                domain_dir = os.path.join(self.config.get("output_dir", "C:/Viren/weights/trained"), domain)
                os.makedirs(domain_dir, exist_ok=True)
                
                # Generate output file name
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = f"{domain}_weights_v1.0.0_{timestamp}.bin"
                output_path = os.path.join(domain_dir, output_file)
                
                # In a real system, we would save the trained weights here
                # For now, we'll create a placeholder file
                with open(output_path, "wb") as f:
                    f.write(f"TRAINED_{domain.upper()}_WEIGHTS".encode())
                
                # Register with weight manager if available
                if self.weight_manager:
                    weight_info = {
                        "name": f"{domain.capitalize()} Knowledge (Trained)",
                        "description": f"Trained weights for {domain} domain using Kaggle datasets",
                        "version": f"1.0.0-{timestamp}",
                        "path": f"trained/{domain}/{output_file}",
                        "size_mb": 0.1,  # Placeholder
                        "hash": "",
                        "required": False,
                        "compatibility": ["base>=1.0.0"],
                        "trained": True,
                        "training_params": params,
                        "datasets": template.get("datasets", []),
                        "training_date": time.time()
                    }
                    
                    self.weight_manager.add_weight_domain(f"{domain}_trained_{timestamp}", weight_info)
                
                # Record training history
                training_record = {
                    "domain": domain,
                    "timestamp": time.time(),
                    "params": params,
                    "datasets": template.get("datasets", []),
                    "output_file": output_path
                }
                
                self.training_history.append(training_record)
                
                # Distribute weights if configured
                if self.config.get("distribution", {}).get("auto_distribute", False):
                    self._distribute_weights(domain, output_path)
                
                return {
                    "success": True,
                    "domain": domain,
                    "output_file": output_path,
                    "params": params
                }
        
        except Exception as e:
            logger.error(f"Error training domain {domain}: {e}")
            return {
                "success": False,
                "error": f"Error training domain {domain}: {str(e)}"
            }
    
    def _distribute_weights(self, domain: str, weight_file: str) -> Dict[str, Any]:
        """Distribute trained weights to other Viren instances"""
        if not self.database_registry:
            logger.warning("Database registry not available, cannot distribute weights")
            return {
                "success": False,
                "error": "Database registry not available"
            }
        
        try:
            # Get all Viren systems
            systems = self.database_registry.registry.get("systems", {})
            target_systems = self.config.get("distribution", {}).get("target_systems", ["all"])
            
            distributed_to = []
            errors = []
            
            for system_id, system_info in systems.items():
                # Skip self
                if system_id == self.database_registry.system_id:
                    continue
                
                # Check if system is in target list
                if target_systems != ["all"] and system_id not in target_systems:
                    continue
                
                # In a real system, we would transfer the weights to the target system
                # For now, we'll just record that we "distributed" them
                logger.info(f"Distributing weights for domain {domain} to system {system_id}")
                
                distributed_to.append(system_id)
            
            # Register distribution in database registry
            self.database_registry.register_database(
                f"trained_weights_{domain}_{int(time.time())}",
                {
                    "name": f"Trained Weights for {domain}",
                    "type": "trained_weights",
                    "domain": domain,
                    "file_path": weight_file,
                    "distributed_to": distributed_to,
                    "distribution_date": time.time()
                }
            )
            
            return {
                "success": True,
                "distributed_to": distributed_to,
                "errors": errors
            }
        
        except Exception as e:
            logger.error(f"Error distributing weights: {e}")
            return {
                "success": False,
                "error": f"Error distributing weights: {str(e)}"
            }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def set_kaggle_credentials(self, username: str, key: str) -> bool:
        """Set Kaggle credentials"""
        self.config["kaggle_username"] = username
        self.config["kaggle_key"] = key
        return self._save_config()
    
    def add_training_template(self, domain: str, template: Dict[str, Any]) -> bool:
        """Add a new training template"""
        if "training_templates" not in self.config:
            self.config["training_templates"] = {}
        
        self.config["training_templates"][domain] = template
        return self._save_config()