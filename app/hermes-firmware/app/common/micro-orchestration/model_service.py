# Systems/services/model_service.py
# Purpose: Model service for Viren

import os
import sys
import logging
import threading
import time
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("model_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ModelService:
    """Model service for Viren."""
    
    def __init__(self):
        """Initialize the model service."""
        self.models = {}
        self.active_models = {}
        self.running = False
        
        # Load model configuration
        try:
            from config.model_config import load_model_config
            self.config = load_model_config()
        except Exception as e:
            logger.error(f"Error loading model configuration: {e}")
            self.config = {
                "bootstrap_model": "gemma:2b",
                "primary_models": {},
                "lightweight_models": {}
            }
    
    def start(self):
        """Start the model service."""
        if self.running:
            logger.warning("Model service already running")
            return True
        
        self.running = True
        
        # Start the bootstrap model
        success = self._start_bootstrap_model()
        if not success:
            logger.error("Failed to start bootstrap model")
            self.running = False
            return False
        
        # Start background model loading
        threading.Thread(target=self._load_models_in_background, daemon=True).start()
        
        logger.info("Model service started")
        return True
    
    def stop(self):
        """Stop the model service."""
        if not self.running:
            logger.warning("Model service not running")
            return True
        
        self.running = False
        logger.info("Model service stopped")
        return True
    
    def _start_bootstrap_model(self):
        """Start the bootstrap model."""
        bootstrap_model = self.config.get("bootstrap_model", "gemma:2b")
        logger.info(f"Starting bootstrap model: {bootstrap_model}")
        
        try:
            import ollama
            
            # Check if model is available
            models = ollama.list()
            model_available = any(model["name"].startswith(bootstrap_model) for model in models["models"])
            
            if not model_available:
                logger.info(f"Downloading bootstrap model: {bootstrap_model}")
                ollama.pull(bootstrap_model)
            
            # Test the model
            response = ollama.generate(
                model=bootstrap_model,
                prompt="You are Viren, an advanced AI assistant. Say hello!",
                stream=False
            )
            
            logger.info(f"Bootstrap model response: {response['response'][:50]}...")
            
            # Add to active models
            self.active_models["bootstrap"] = bootstrap_model
            
            return True
        except Exception as e:
            logger.error(f"Error starting bootstrap model: {e}")
            return False
    
    def _load_models_in_background(self):
        """Load models in background."""
        try:
            import ollama
            
            # Get download priority
            download_priority = self.config.get("download_priority", [])
            
            # Skip the first model (bootstrap)
            if download_priority and download_priority[0] == self.config.get("bootstrap_model"):
                download_priority = download_priority[1:]
            
            # Download models
            for model in download_priority:
                if not self.running:
                    break
                
                try:
                    logger.info(f"Downloading model: {model}")
                    ollama.pull(model)
                    logger.info(f"Downloaded model: {model}")
                except Exception as e:
                    logger.error(f"Error downloading model {model}: {e}")
                
                # Sleep to avoid overloading
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error loading models in background: {e}")
    
    def get_model(self, task: str) -> str:
        """Get a model for a task."""
        from config.model_config import select_model_for_task
        return select_model_for_task(task, self.config)
    
    def generate(self, prompt: str, task: str = None, model: str = None, **kwargs) -> str:
        """Generate text with a model."""
        if not self.running:
            logger.warning("Model service not running")
            return "Model service not running"
        
        # Select model
        if model is None:
            if task is not None:
                model = self.get_model(task)
            else:
                model = self.config.get("bootstrap_model", "gemma:2b")
        
        try:
            import ollama
            
            # Generate text
            response = ollama.generate(
                model=model,
                prompt=prompt,
                stream=False,
                **kwargs
            )
            
            return response["response"]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

# Create singleton instance
model_service = ModelService()

def start_service():
    """Start the model service."""
    return model_service.start()

if __name__ == "__main__":
    start_service()
    
    # Keep the service running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        model_service.stop()