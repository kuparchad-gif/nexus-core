#!/usr/bin/env python3
"""
AI Studio Integration for Cloud Viren
Handles model loading and inference through AI Studio
"""

import os
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AIStudioIntegration")

class AIStudioClient:
    """
    Client for AI Studio integration
    Handles model loading, inference, and management
    """
    
    def __init__(self, api_key: str = None, config_path: str = None):
        """Initialize the AI Studio client"""
        self.api_key = api_key or os.environ.get("AI_STUDIO_API_KEY")
        self.config_path = config_path or os.path.join("config", "ai_studio_config.json")
        self.base_url = "https://api.ai-studio.com/v1"  # Example URL
        self.loaded_models = {}
        self.model_cache = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load AI Studio configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key") or self.api_key
                    self.base_url = config.get("base_url") or self.base_url
                    self.loaded_models = config.get("loaded_models", {})
                    logger.info("Loaded AI Studio configuration")
            except Exception as e:
                logger.error(f"Error loading AI Studio configuration: {e}")
    
    def _save_config(self):
        """Save AI Studio configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            config = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "loaded_models": self.loaded_models
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Saved AI Studio configuration")
        except Exception as e:
            logger.error(f"Error saving AI Studio configuration: {e}")
    
    def load_model(self, model_id: str, model_size: str) -> Dict[str, Any]:
        """
        Load a model in AI Studio
        
        Args:
            model_id: Model identifier (e.g., "google/gemma-3-3b-it")
            model_size: Size category (e.g., "3B")
            
        Returns:
            Dictionary with load status and information
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Loading model {model_id} ({model_size})")
        
        try:
            # Check if model is already loaded
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} already loaded")
                return {
                    "status": "success",
                    "message": "Model already loaded",
                    "model_id": model_id,
                    "model_size": model_size,
                    "model_info": self.loaded_models[model_id]
                }
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model_id": model_id,
                "parameters": {
                    "load_in_8bit": model_size in ["1B", "3B", "7B"],
                    "load_in_4bit": model_size in ["14B", "27B"],
                    "use_flash_attention": True
                }
            }
            
            # Send request to load model
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/models/load",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                load_time = time.time() - start_time
                
                # Update loaded models
                self.loaded_models[model_id] = {
                    "model_size": model_size,
                    "load_time": load_time,
                    "loaded_at": time.time(),
                    "model_info": result.get("model_info", {})
                }
                
                # Save configuration
                self._save_config()
                
                logger.info(f"Model {model_id} loaded successfully in {load_time:.2f} seconds")
                return {
                    "status": "success",
                    "message": "Model loaded successfully",
                    "model_id": model_id,
                    "model_size": model_size,
                    "load_time": load_time
                }
            else:
                logger.error(f"Error loading model: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error loading model: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {"status": "error", "message": str(e)}
    
    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """
        Unload a model from AI Studio
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with unload status
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Unloading model {model_id}")
        
        try:
            # Check if model is loaded
            if model_id not in self.loaded_models:
                logger.info(f"Model {model_id} not loaded")
                return {
                    "status": "success",
                    "message": "Model not loaded"
                }
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model_id": model_id
            }
            
            # Send request to unload model
            response = requests.post(
                f"{self.base_url}/models/unload",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                # Remove from loaded models
                del self.loaded_models[model_id]
                
                # Save configuration
                self._save_config()
                
                logger.info(f"Model {model_id} unloaded successfully")
                return {
                    "status": "success",
                    "message": "Model unloaded successfully"
                }
            else:
                logger.error(f"Error unloading model: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error unloading model: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_text(self, model_id: str, prompt: str, 
                     max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text using a loaded model
        
        Args:
            model_id: Model identifier
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary with generated text and metadata
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Generating text with model {model_id}")
        
        try:
            # Check if model is loaded
            if model_id not in self.loaded_models:
                logger.warning(f"Model {model_id} not loaded, attempting to load")
                load_result = self.load_model(model_id, "unknown")
                if load_result["status"] != "success":
                    return load_result
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Send request for text generation
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                generation_time = time.time() - start_time
                
                generated_text = result.get("choices", [{}])[0].get("text", "")
                
                logger.info(f"Text generated successfully in {generation_time:.2f} seconds")
                return {
                    "status": "success",
                    "text": generated_text,
                    "generation_time": generation_time,
                    "model_id": model_id,
                    "usage": result.get("usage", {})
                }
            else:
                logger.error(f"Error generating text: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error generating text: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"status": "error", "message": str(e)}
    
    def process_with_cascade(self, prompt: str, model_cascade: Dict[str, Dict[str, Any]],
                           start_level: str = "1B") -> Dict[str, Any]:
        """
        Process text through a model cascade
        
        Args:
            prompt: Input prompt
            model_cascade: Dictionary of model cascade configuration
            start_level: Starting level in the cascade
            
        Returns:
            Dictionary with final output and processing steps
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        if start_level not in model_cascade:
            return {"status": "error", "message": f"Unknown start level: {start_level}"}
        
        logger.info(f"Processing with cascade starting at {start_level}")
        
        current_level = start_level
        current_output = prompt
        processing_steps = []
        
        while current_level:
            try:
                model_info = model_cascade[current_level]
                model_id = model_info["model_id"]
                
                # Generate text with current model
                result = self.generate_text(
                    model_id=model_id,
                    prompt=current_output,
                    max_tokens=1024,
                    temperature=0.7
                )
                
                if result["status"] != "success":
                    processing_steps.append({
                        "level": current_level,
                        "model_id": model_id,
                        "error": result["message"]
                    })
                    break
                
                # Update current output
                current_output = result["text"]
                
                # Record processing step
                processing_steps.append({
                    "level": current_level,
                    "model_id": model_id,
                    "generation_time": result["generation_time"],
                    "tokens": result.get("usage", {}).get("total_tokens", 0)
                })
                
                logger.info(f"Processed with {current_level} ({model_id})")
                
                # Move to next level or finish
                current_level = model_info.get("next_level")
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
            "processing_steps": processing_steps
        }
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with loaded models information
        """
        return {
            "status": "success",
            "loaded_models": self.loaded_models
        }

# Example usage
if __name__ == "__main__":
    # Create AI Studio client
    client = AIStudioClient()
    
    # Example model cascade
    MODEL_CASCADE = {
        "1B": {
            "model_id": "google/gemma-1.1-1b-it",
            "next_level": "3B"
        },
        "3B": {
            "model_id": "google/gemma-3-3b-it",
            "next_level": "7B"
        },
        "7B": {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "next_level": None
        }
    }
    
    # Load a model
    result = client.load_model("google/gemma-1.1-1b-it", "1B")
    print(f"Load result: {result}")
    
    # Generate text
    result = client.generate_text(
        model_id="google/gemma-1.1-1b-it",
        prompt="Hello, I am"
    )
    print(f"Generation result: {result}")
    
    # Process with cascade
    result = client.process_with_cascade(
        prompt="Explain quantum computing",
        model_cascade=MODEL_CASCADE,
        start_level="1B"
    )
    print(f"Cascade result: {result}")