#!/usr/bin/env python3
"""
Model Cascade Manager for Cloud Viren
Manages the model cascade from 1B to 256B models
"""

import os
import sys
import json
import time
import logging
import threading
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelCascadeManager")

class ModelCascadeManager:
    """
    Model Cascade Manager for Cloud Viren
    Manages the model cascade from 1B to 256B models
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the model cascade manager"""
        self.config_path = config_path or os.path.join("config", "model_cascade_config.json")
        self.config = self._load_config()
        self.model_status = {}
        self.cascade_status = "inactive"
        self.active_models = set()
        self.model_locks = {}
        self.model_usage = {}
        self.model_metrics = {}
        self.download_queue = []
        self.download_thread = None
        self.is_downloading = False
        
        # Initialize model locks
        for model_id in self.config["models"]:
            self.model_locks[model_id] = threading.Lock()
        
        logger.info(f"Model Cascade Manager initialized with {len(self.config['models'])} models")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "models": {
                "1B": {
                    "model_id": "google/gemma-1.1-1b-it",
                    "provider": "ai_studio",
                    "next_level": "3B",
                    "priority": 1,
                    "auto_download": True
                },
                "3B": {
                    "model_id": "google/gemma-3-3b-it",
                    "provider": "ai_studio",
                    "next_level": "7B",
                    "priority": 2,
                    "auto_download": True
                },
                "7B": {
                    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "provider": "ai_studio",
                    "next_level": "14B",
                    "priority": 3,
                    "auto_download": True
                },
                "14B": {
                    "model_id": "google/gemma-3-14b-it",
                    "provider": "ai_studio",
                    "next_level": "27B",
                    "priority": 4,
                    "auto_download": False
                },
                "27B": {
                    "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
                    "provider": "ai_studio",
                    "next_level": "128B",
                    "priority": 5,
                    "auto_download": False
                },
                "128B": {
                    "model_id": "anthropic/claude-3-opus",
                    "provider": "ai_studio",
                    "next_level": "256B",
                    "priority": 6,
                    "auto_download": False
                },
                "256B": {
                    "model_id": "anthropic/claude-3-sonnet",
                    "provider": "ai_studio",
                    "next_level": None,
                    "priority": 7,
                    "auto_download": False
                }
            },
            "specialized_models": {
                "sql": {
                    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "provider": "ai_studio",
                    "fine_tuned": "sql",
                    "priority": 3,
                    "auto_download": True
                },
                "code": {
                    "model_id": "codellama/CodeLlama-7b-Instruct",
                    "provider": "ai_studio",
                    "priority": 3,
                    "auto_download": True
                },
                "vision": {
                    "model_id": "llava-hf/llava-1.5-7b-hf",
                    "provider": "hyperbolic",
                    "priority": 3,
                    "auto_download": True
                }
            },
            "cascade_settings": {
                "auto_cascade": True,
                "confidence_threshold": 0.7,
                "max_cascade_depth": 3,
                "cascade_timeout": 60,  # seconds
                "cascade_batch_size": 5
            },
            "download_settings": {
                "concurrent_downloads": 1,
                "retry_attempts": 3,
                "retry_delay": 5,  # seconds
                "download_timeout": 3600  # 1 hour
            },
            "providers": {
                "ai_studio": {
                    "endpoint": "https://api.ai-studio.com/v1",
                    "api_key": ""
                },
                "hyperbolic": {
                    "endpoint": "https://api.hyperbolic.ai/v1",
                    "api_key": ""
                }
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config.get(key), dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    
                    logger.info("Model cascade configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading model cascade configuration: {e}")
        
        logger.info("Using default model cascade configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Model cascade configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving model cascade configuration: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize the model cascade"""
        logger.info("Initializing model cascade")
        
        try:
            # Check model status
            self._check_model_status()
            
            # Start download thread if needed
            if self.download_queue and not self.download_thread:
                self.download_thread = threading.Thread(target=self._download_models_loop)
                self.download_thread.daemon = True
                self.download_thread.start()
            
            # Set cascade status
            if any(status.get("status") == "ready" for status in self.model_status.values()):
                self.cascade_status = "ready"
            else:
                self.cascade_status = "initializing"
            
            logger.info(f"Model cascade initialized with status: {self.cascade_status}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing model cascade: {e}")
            self.cascade_status = "error"
            return False
    
    def _check_model_status(self) -> None:
        """Check status of all models"""
        # Check regular models
        for model_size, model_info in self.config["models"].items():
            model_id = model_info["model_id"]
            
            # Check if model is downloaded
            is_downloaded = self._is_model_downloaded(model_id)
            
            # Update model status
            self.model_status[model_id] = {
                "model_size": model_size,
                "provider": model_info["provider"],
                "downloaded": is_downloaded,
                "status": "ready" if is_downloaded else "not_downloaded",
                "last_check": time.time()
            }
            
            # Add to download queue if needed
            if not is_downloaded and model_info.get("auto_download", False):
                self._add_to_download_queue(model_id, model_info)
        
        # Check specialized models
        for model_type, model_info in self.config["specialized_models"].items():
            model_id = model_info["model_id"]
            
            # Check if model is downloaded
            is_downloaded = self._is_model_downloaded(model_id)
            
            # Update model status
            self.model_status[model_id] = {
                "model_type": model_type,
                "provider": model_info["provider"],
                "downloaded": is_downloaded,
                "status": "ready" if is_downloaded else "not_downloaded",
                "last_check": time.time()
            }
            
            # Add to download queue if needed
            if not is_downloaded and model_info.get("auto_download", False):
                self._add_to_download_queue(model_id, model_info)
    
    def _is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded"""
        # This is a simplified implementation
        # In a real implementation, you would check if the model files exist
        
        # For now, just check if the model directory exists
        model_dir = os.path.join("models", model_id.replace("/", "_"))
        return os.path.exists(model_dir)
    
    def _add_to_download_queue(self, model_id: str, model_info: Dict[str, Any]) -> None:
        """Add a model to the download queue"""
        # Check if model is already in queue
        for item in self.download_queue:
            if item["model_id"] == model_id:
                return
        
        # Add to queue
        self.download_queue.append({
            "model_id": model_id,
            "provider": model_info["provider"],
            "priority": model_info.get("priority", 999),
            "added_time": time.time()
        })
        
        # Sort queue by priority
        self.download_queue.sort(key=lambda x: x["priority"])
        
        logger.info(f"Added model {model_id} to download queue")
    
    def _download_models_loop(self) -> None:
        """Download models from the queue"""
        logger.info("Starting model download loop")
        
        while self.download_queue:
            try:
                # Get next model to download
                model_item = self.download_queue[0]
                model_id = model_item["model_id"]
                provider = model_item["provider"]
                
                # Set downloading flag
                self.is_downloading = True
                
                # Update model status
                if model_id in self.model_status:
                    self.model_status[model_id]["status"] = "downloading"
                
                # Download model
                logger.info(f"Downloading model {model_id} from {provider}")
                success = self._download_model(model_id, provider)
                
                if success:
                    logger.info(f"Model {model_id} downloaded successfully")
                    
                    # Update model status
                    if model_id in self.model_status:
                        self.model_status[model_id]["downloaded"] = True
                        self.model_status[model_id]["status"] = "ready"
                        self.model_status[model_id]["last_check"] = time.time()
                else:
                    logger.error(f"Failed to download model {model_id}")
                    
                    # Update model status
                    if model_id in self.model_status:
                        self.model_status[model_id]["status"] = "download_failed"
                
                # Remove from queue
                self.download_queue.pop(0)
            
            except Exception as e:
                logger.error(f"Error in download loop: {e}")
                time.sleep(10)  # Wait before retrying
            
            finally:
                # Clear downloading flag
                self.is_downloading = False
        
        logger.info("Download queue empty, stopping download loop")
    
    def _download_model(self, model_id: str, provider: str) -> bool:
        """Download a model"""
        # This is a simplified implementation
        # In a real implementation, you would download the model files
        
        try:
            # Create model directory
            model_dir = os.path.join("models", model_id.replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a placeholder file
            with open(os.path.join(model_dir, "downloaded.txt"), 'w') as f:
                f.write(f"Model {model_id} from {provider}\n")
                f.write(f"Downloaded at {time.time()}\n")
            
            # Simulate download time
            time.sleep(2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            return False
    
    def get_model(self, model_size: str = None, model_type: str = None) -> Optional[Dict[str, Any]]:
        """Get a model by size or type"""
        if model_size:
            # Get model by size
            if model_size in self.config["models"]:
                model_info = self.config["models"][model_size]
                model_id = model_info["model_id"]
                
                # Check if model is ready
                if model_id in self.model_status and self.model_status[model_id]["status"] == "ready":
                    return {
                        "model_id": model_id,
                        "model_size": model_size,
                        "provider": model_info["provider"],
                        "next_level": model_info["next_level"]
                    }
            
            return None
        
        elif model_type:
            # Get model by type
            if model_type in self.config["specialized_models"]:
                model_info = self.config["specialized_models"][model_type]
                model_id = model_info["model_id"]
                
                # Check if model is ready
                if model_id in self.model_status and self.model_status[model_id]["status"] == "ready":
                    return {
                        "model_id": model_id,
                        "model_type": model_type,
                        "provider": model_info["provider"],
                        "fine_tuned": model_info.get("fine_tuned")
                    }
            
            return None
        
        else:
            # Get default model (smallest available)
            for model_size in sorted(self.config["models"].keys(), key=lambda x: self.config["models"][x]["priority"]):
                model_info = self.config["models"][model_size]
                model_id = model_info["model_id"]
                
                # Check if model is ready
                if model_id in self.model_status and self.model_status[model_id]["status"] == "ready":
                    return {
                        "model_id": model_id,
                        "model_size": model_size,
                        "provider": model_info["provider"],
                        "next_level": model_info["next_level"]
                    }
            
            return None
    
    def process_with_cascade(self, input_text: str, start_level: str = "1B", max_depth: int = None) -> Dict[str, Any]:
        """Process text through the model cascade"""
        logger.info(f"Processing input with cascade starting at {start_level}")
        
        # Set default max depth
        if max_depth is None:
            max_depth = self.config["cascade_settings"]["max_cascade_depth"]
        
        # Check if cascade is ready
        if self.cascade_status != "ready":
            return {
                "status": "error",
                "message": f"Model cascade is not ready: {self.cascade_status}"
            }
        
        # Check if start level exists
        if start_level not in self.config["models"]:
            return {
                "status": "error",
                "message": f"Unknown model size: {start_level}"
            }
        
        # Initialize cascade
        current_level = start_level
        current_output = input_text
        processing_steps = []
        depth = 0
        
        # Process through cascade
        while current_level and depth < max_depth:
            try:
                # Get model info
                model_info = self.config["models"][current_level]
                model_id = model_info["model_id"]
                provider = model_info["provider"]
                
                # Check if model is ready
                if model_id not in self.model_status or self.model_status[model_id]["status"] != "ready":
                    logger.warning(f"Model {model_id} is not ready, stopping cascade")
                    break
                
                # Process with current model
                logger.info(f"Processing with {current_level} ({model_id})")
                
                # Acquire model lock
                with self.model_locks[model_id]:
                    # Process input
                    result = self._process_with_model(model_id, provider, current_output)
                
                if result["status"] == "success":
                    # Update current output
                    current_output = result["output"]
                    
                    # Record processing step
                    processing_steps.append({
                        "level": current_level,
                        "model_id": model_id,
                        "provider": provider,
                        "processing_time": result["processing_time"],
                        "confidence": result.get("confidence", 0.0)
                    })
                    
                    # Update model usage
                    self._update_model_usage(model_id)
                    
                    # Check if we should continue cascade
                    if self._should_continue_cascade(result):
                        # Move to next level
                        current_level = model_info["next_level"]
                        depth += 1
                    else:
                        # Stop cascade
                        break
                else:
                    # Error processing with model
                    processing_steps.append({
                        "level": current_level,
                        "model_id": model_id,
                        "provider": provider,
                        "error": result["message"]
                    })
                    
                    # Try next level if available
                    current_level = model_info["next_level"]
                    depth += 1
            
            except Exception as e:
                logger.error(f"Error in cascade at level {current_level}: {e}")
                processing_steps.append({
                    "level": current_level,
                    "error": str(e)
                })
                break
        
        return {
            "status": "success",
            "final_output": current_output,
            "processing_steps": processing_steps,
            "cascade_depth": depth,
            "start_level": start_level,
            "end_level": current_level
        }
    
    def _process_with_model(self, model_id: str, provider: str, input_text: str) -> Dict[str, Any]:
        """Process text with a specific model"""
        # This is a simplified implementation
        # In a real implementation, you would use the actual model
        
        try:
            # Simulate processing time
            processing_time = 1.0 + len(input_text) / 1000
            time.sleep(processing_time)
            
            # Generate output
            output = f"Processed by {model_id}: {input_text}"
            
            # Generate random confidence
            import random
            confidence = random.uniform(0.7, 0.95)
            
            return {
                "status": "success",
                "output": output,
                "processing_time": processing_time,
                "confidence": confidence
            }
        
        except Exception as e:
            logger.error(f"Error processing with model {model_id}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _should_continue_cascade(self, result: Dict[str, Any]) -> bool:
        """Check if cascade should continue to next level"""
        # Check if auto cascade is enabled
        if not self.config["cascade_settings"]["auto_cascade"]:
            return False
        
        # Check confidence threshold
        confidence = result.get("confidence", 0.0)
        threshold = self.config["cascade_settings"]["confidence_threshold"]
        
        return confidence < threshold
    
    def _update_model_usage(self, model_id: str) -> None:
        """Update model usage statistics"""
        if model_id not in self.model_usage:
            self.model_usage[model_id] = {
                "count": 0,
                "last_used": 0
            }
        
        self.model_usage[model_id]["count"] += 1
        self.model_usage[model_id]["last_used"] = time.time()
    
    def get_model_status(self, model_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of a specific model or all models"""
        if model_id:
            return self.model_status.get(model_id, {"status": "unknown"})
        else:
            return [
                {
                    "model_id": model_id,
                    **status
                }
                for model_id, status in self.model_status.items()
            ]
    
    def get_cascade_status(self) -> Dict[str, Any]:
        """Get status of the model cascade"""
        return {
            "status": self.cascade_status,
            "models": len(self.model_status),
            "ready_models": sum(1 for status in self.model_status.values() if status.get("status") == "ready"),
            "downloading": self.is_downloading,
            "download_queue": len(self.download_queue)
        }
    
    def get_model_usage(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        return self.model_usage

# Example usage
if __name__ == "__main__":
    # Create model cascade manager
    manager = ModelCascadeManager()
    
    # Initialize cascade
    manager.initialize()
    
    # Print cascade status
    print(f"Cascade status: {manager.get_cascade_status()}")
    
    # Print model status
    for model in manager.get_model_status():
        print(f"Model {model['model_id']}: {model['status']}")
    
    # Process with cascade
    result = manager.process_with_cascade("Hello, world!", start_level="1B")
    print(f"Cascade result: {result}")