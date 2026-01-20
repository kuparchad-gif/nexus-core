#!/usr/bin/env python3
"""
Model Registry for Cloud Viren
Tracks and manages available models
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelRegistry")

class ModelRegistry:
    """Registry for tracking and managing models"""
    
    def __init__(self, models_dir=None, registry_path=None):
        """Initialize the model registry"""
        self.models_dir = models_dir or "/app/models"
        self.registry_path = registry_path or os.path.join(self.models_dir, "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load the model registry"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default registry
                default_registry = {
                    "models": {},
                    "last_updated": 0
                }
                self._save_registry(default_registry)
                return default_registry
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            return {"models": {}, "last_updated": 0}
    
    def _save_registry(self, registry):
        """Save the model registry"""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
            return False
    
    def scan_models(self):
        """Scan for available models"""
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                logger.warning(f"Models directory not found: {self.models_dir}")
                return False
            
            # Scan for model directories
            for item in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, item)
                
                # Skip non-directories and registry file
                if not os.path.isdir(model_path) or item == "model_registry.json":
                    continue
                
                # Check if model is already in registry
                if item not in self.registry["models"]:
                    self.registry["models"][item] = {
                        "name": item,
                        "path": model_path,
                        "type": self._detect_model_type(model_path),
                        "loaded": False,
                        "last_checked": int(os.path.getmtime(model_path))
                    }
                    logger.info(f"Added model to registry: {item}")
                else:
                    # Update last checked time
                    self.registry["models"][item]["last_checked"] = int(os.path.getmtime(model_path))
            
            # Save updated registry
            import time
            self.registry["last_updated"] = int(time.time())
            self._save_registry(self.registry)
            
            return True
        
        except Exception as e:
            logger.error(f"Error scanning models: {e}")
            return False
    
    def _detect_model_type(self, model_path):
        """Detect the type of model based on files"""
        try:
            files = os.listdir(model_path)
            
            # Check for common model file patterns
            if any(f.endswith(".gguf") for f in files):
                return "gguf"
            elif any(f.endswith(".safetensors") for f in files):
                return "safetensors"
            elif any(f == "model.bin" for f in files) or any(f.endswith(".bin") for f in files):
                return "pytorch"
            elif any(f == "saved_model.pb" for f in files):
                return "tensorflow"
            else:
                return "unknown"
        
        except Exception:
            return "unknown"
    
    def get_models(self):
        """Get all registered models"""
        return self.registry["models"]
    
    def get_model(self, model_id):
        """Get a specific model"""
        return self.registry["models"].get(model_id)
    
    def mark_model_loaded(self, model_id, loaded=True):
        """Mark a model as loaded or unloaded"""
        if model_id in self.registry["models"]:
            self.registry["models"][model_id]["loaded"] = loaded
            self._save_registry(self.registry)
            return True
        return False