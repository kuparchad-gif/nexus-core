#!/usr/bin/env python3
"""
Model Manager for Viren Platinum Edition
Handles model loading, switching, and inference across different providers
"""

import os
import json
import time
import logging
import requests
import subprocess
import threading
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger("ModelManager")

class ModelManager:
    """
    Manages AI models across different providers (Ollama, vLLM, LM Studio)
    Supports hot-swapping and automatic model discovery/download
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the model manager"""
        self.config_path = config_path or os.path.join("config", "model_config.json")
        self.models = {}
        self.providers = {
            "Ollama": {
                "api_url": "http://localhost:11434/api",
                "available": self._check_ollama()
            },
            "vLLM": {
                "api_url": "http://localhost:8000/v1",
                "available": self._check_vllm()
            },
            "LM Studio": {
                "api_url": "http://localhost:1234/v1",
                "available": self._check_lmstudio()
            },
            "API": {
                "api_url": "https://api.openai.com/v1",
                "available": True  # Assume API is always available
            }
        }
        self.active_model = None
        self.active_provider = None
        self.model_metrics = {}
        
        # Load configuration
        self._load_config()
        
        # Set default model
        self._set_default_model()
    
    def _load_config(self):
        """Load model configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.models = config.get("models", {})
                    logger.info(f"Loaded {len(self.models)} models from configuration")
            except Exception as e:
                logger.error(f"Error loading model configuration: {e}")
                self._create_default_config()
        else:
            logger.info("Model configuration not found, creating default")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self.models = {
            "gemma:2b": {
                "size": "700MB",
                "type": "general",
                "provider": "Ollama",
                "description": "Fast, lightweight model for quick responses"
            },
            "hermes:2-pro-llama-3-7b": {
                "size": "4GB",
                "type": "advanced",
                "provider": "Ollama",
                "description": "Advanced reasoning and problem-solving"
            },
            "phi:3-mini-4k": {
                "size": "1.3GB",
                "type": "efficient",
                "provider": "Ollama",
                "description": "Efficient model for portable use"
            },
            "codellama:7b": {
                "size": "4GB",
                "type": "code",
                "provider": "Ollama",
                "description": "Specialized for code generation"
            }
        }
        
        # Save default configuration
        self._save_config()
    
    def _save_config(self):
        """Save model configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, 'w') as f:
                json.dump({"models": self.models}, f, indent=2)
            logger.info("Model configuration saved")
        except Exception as e:
            logger.error(f"Error saving model configuration: {e}")
    
    def _set_default_model(self):
        """Set default model based on availability"""
        # Try to find an available model
        for model_name, model_info in self.models.items():
            provider = model_info.get("provider", "Ollama")
            if self.providers.get(provider, {}).get("available", False):
                self.set_active_model(model_name)
                return
        
        # If no model is available, set the first one anyway
        if self.models:
            first_model = next(iter(self.models.keys()))
            self.active_model = first_model
            self.active_provider = self.models[first_model].get("provider", "Ollama")
            logger.warning(f"Set default model to {first_model} but provider may not be available")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_vllm(self) -> bool:
        """Check if vLLM is available"""
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_lmstudio(self) -> bool:
        """Check if LM Studio is available"""
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self, provider: str = None) -> List[str]:
        """
        Get available models, optionally filtered by provider
        
        Args:
            provider: Provider name or None for all
            
        Returns:
            List of available model names
        """
        if provider:
            # Check if provider is available
            if not self.providers.get(provider, {}).get("available", False):
                logger.warning(f"Provider {provider} is not available")
                return []
            
            # Get models for this provider
            return self._get_provider_models(provider)
        else:
            # Get all models from all available providers
            all_models = []
            for provider_name, provider_info in self.providers.items():
                if provider_info.get("available", False):
                    all_models.extend(self._get_provider_models(provider_name))
            return all_models
    
    def _get_provider_models(self, provider: str) -> List[str]:
        """
        Get models from a specific provider
        
        Args:
            provider: Provider name
            
        Returns:
            List of model names
        """
        if provider == "Ollama":
            return self._get_ollama_models()
        elif provider == "vLLM":
            return self._get_vllm_models()
        elif provider == "LM Studio":
            return self._get_lmstudio_models()
        elif provider == "API":
            return self._get_api_models()
        else:
            return []
    
    def _get_ollama_models(self) -> List[str]:
        """Get models from Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    def _get_vllm_models(self) -> List[str]:
        """Get models from vLLM"""
        try:
            response = requests.get("http://localhost:8000/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting vLLM models: {e}")
            return []
    
    def _get_lmstudio_models(self) -> List[str]:
        """Get models from LM Studio"""
        try:
            response = requests.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting LM Studio models: {e}")
            return []
    
    def _get_api_models(self) -> List[str]:
        """Get models from API"""
        # For API, we return a predefined list
        return ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
    
    def set_active_model(self, model_name: str, provider: str = None) -> bool:
        """
        Set the active model
        
        Args:
            model_name: Model name
            provider: Provider name or None to use the model's default provider
            
        Returns:
            True if successful, False otherwise
        """
        # If provider is not specified, use the model's default provider
        if not provider:
            if model_name in self.models:
                provider = self.models[model_name].get("provider", "Ollama")
            else:
                provider = "Ollama"
        
        # Check if provider is available
        if not self.providers.get(provider, {}).get("available", False):
            logger.warning(f"Provider {provider} is not available")
            return False
        
        # Check if model exists for this provider
        available_models = self._get_provider_models(provider)
        if model_name not in available_models:
            # Try to download the model
            logger.info(f"Model {model_name} not found, attempting to download")
            if not self.download_model(model_name, provider):
                logger.error(f"Failed to download model {model_name}")
                return False
        
        # Set active model and provider
        self.active_model = model_name
        self.active_provider = provider
        logger.info(f"Active model set to {model_name} with provider {provider}")
        
        # Add to models if not already present
        if model_name not in self.models:
            self.models[model_name] = {
                "provider": provider,
                "size": "Unknown",
                "type": "general",
                "description": "Automatically added model"
            }
            self._save_config()
        
        return True
    
    def download_model(self, model_name: str, provider: str) -> bool:
        """
        Download a model
        
        Args:
            model_name: Model name
            provider: Provider name
            
        Returns:
            True if successful, False otherwise
        """
        if provider == "Ollama":
            return self._download_ollama_model(model_name)
        elif provider == "vLLM":
            return self._download_vllm_model(model_name)
        elif provider == "LM Studio":
            return self._download_lmstudio_model(model_name)
        else:
            logger.error(f"Download not supported for provider {provider}")
            return False
    
    def _download_ollama_model(self, model_name: str) -> bool:
        """Download a model from Ollama"""
        try:
            logger.info(f"Downloading Ollama model: {model_name}")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Successfully downloaded Ollama model: {model_name}")
                return True
            else:
                logger.error(f"Failed to download Ollama model: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error downloading Ollama model: {e}")
            return False
    
    def _download_vllm_model(self, model_name: str) -> bool:
        """Download a model for vLLM"""
        # vLLM typically downloads models on first use
        logger.info(f"vLLM will download model {model_name} on first use")
        return True
    
    def _download_lmstudio_model(self, model_name: str) -> bool:
        """Download a model for LM Studio"""
        # LM Studio has its own UI for downloading models
        logger.info(f"Please use LM Studio UI to download model {model_name}")
        return False
    
    def process_message(self, message: str) -> str:
        """
        Process a message using the active model
        
        Args:
            message: Input message
            
        Returns:
            Model response
        """
        if not self.active_model or not self.active_provider:
            return "No active model set. Please select a model first."
        
        # Record start time for metrics
        start_time = time.time()
        
        try:
            if self.active_provider == "Ollama":
                response = self._process_ollama(message)
            elif self.active_provider == "vLLM":
                response = self._process_vllm(message)
            elif self.active_provider == "LM Studio":
                response = self._process_lmstudio(message)
            elif self.active_provider == "API":
                response = self._process_api(message)
            else:
                return f"Unknown provider: {self.active_provider}"
            
            # Record metrics
            elapsed = time.time() - start_time
            self._update_metrics(elapsed, len(response))
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error: {str(e)}"
    
    def _process_ollama(self, message: str) -> str:
        """Process a message using Ollama"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.active_model,
                    "prompt": message
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.text}")
                return f"Error: {response.text}"
        except Exception as e:
            logger.error(f"Error processing with Ollama: {e}")
            return f"Error: {str(e)}"
    
    def _process_vllm(self, message: str) -> str:
        """Process a message using vLLM"""
        try:
            response = requests.post(
                "http://localhost:8000/v1/completions",
                json={
                    "model": self.active_model,
                    "prompt": message,
                    "max_tokens": 1024
                }
            )
            
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("text", "")
            else:
                logger.error(f"vLLM error: {response.text}")
                return f"Error: {response.text}"
        except Exception as e:
            logger.error(f"Error processing with vLLM: {e}")
            return f"Error: {str(e)}"
    
    def _process_lmstudio(self, message: str) -> str:
        """Process a message using LM Studio"""
        try:
            response = requests.post(
                "http://localhost:1234/v1/completions",
                json={
                    "model": self.active_model,
                    "prompt": message,
                    "max_tokens": 1024
                }
            )
            
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("text", "")
            else:
                logger.error(f"LM Studio error: {response.text}")
                return f"Error: {response.text}"
        except Exception as e:
            logger.error(f"Error processing with LM Studio: {e}")
            return f"Error: {str(e)}"
    
    def _process_api(self, message: str) -> str:
        """Process a message using API"""
        # This is a placeholder for API processing
        return f"API response to: {message}"
    
    def _update_metrics(self, elapsed_time: float, response_length: int):
        """Update metrics for the active model"""
        if not self.active_model:
            return
        
        if self.active_model not in self.model_metrics:
            self.model_metrics[self.active_model] = {
                "response_times": [],
                "token_rates": [],
                "avg_response_time": 0,
                "avg_token_rate": 0,
                "total_calls": 0
            }
        
        metrics = self.model_metrics[self.active_model]
        
        # Approximate tokens as words (response_length / 4)
        tokens = response_length / 4
        token_rate = tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Update metrics
        metrics["response_times"].append(elapsed_time)
        metrics["token_rates"].append(token_rate)
        metrics["total_calls"] += 1
        
        # Keep only the last 10 measurements
        if len(metrics["response_times"]) > 10:
            metrics["response_times"] = metrics["response_times"][-10:]
            metrics["token_rates"] = metrics["token_rates"][-10:]
        
        # Update averages
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
        metrics["avg_token_rate"] = sum(metrics["token_rates"]) / len(metrics["token_rates"])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all models"""
        return self.model_metrics
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        return self.models.get(model_name, {})
    
    def hot_swap_model(self, model_name: str, provider: str = None) -> bool:
        """
        Hot swap to a different model without interrupting the service
        
        Args:
            model_name: Model name
            provider: Provider name or None to use the model's default provider
            
        Returns:
            True if successful, False otherwise
        """
        # This is a simple implementation that just sets the active model
        # In a real implementation, this would handle more complex swapping logic
        return self.set_active_model(model_name, provider)
    
    def discover_models(self) -> List[str]:
        """
        Discover available models from all providers
        
        Returns:
            List of discovered model names
        """
        discovered_models = []
        
        # Check each provider
        for provider_name, provider_info in self.providers.items():
            if provider_info.get("available", False):
                models = self._get_provider_models(provider_name)
                for model in models:
                    discovered_models.append(model)
                    
                    # Add to models if not already present
                    if model not in self.models:
                        self.models[model] = {
                            "provider": provider_name,
                            "size": "Unknown",
                            "type": "general",
                            "description": "Automatically discovered model"
                        }
        
        # Save updated models
        self._save_config()
        
        return discovered_models

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create model manager
    manager = ModelManager()
    
    # Get available models
    models = manager.get_available_models()
    print(f"Available models: {models}")
    
    # Set active model
    if models:
        manager.set_active_model(models[0])
        
        # Process a message
        response = manager.process_message("Hello, how are you?")
        print(f"Response: {response}")
        
        # Get metrics
        metrics = manager.get_metrics()
        print(f"Metrics: {metrics}")
    else:
        print("No models available")