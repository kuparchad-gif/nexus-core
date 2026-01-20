# Systems/bridge/model_router.py - Routes queries to appropriate models

import logging
import os
import json
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger("model_router")
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
handler = logging.FileHandler("logs/model_router.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ModelRouter:
    """Routes queries to appropriate models based on role and context."""
    
    def __init__(self):
        """Initialize the model router."""
        self.backends = {}
        self.models = {}
        self.role_mappings = {}
        self.default_model = None
    
    def initialize_backends(self):
        """Initialize all available backends."""
        try:
            # Try to import available backends
            backends = self._discover_backends()
            
            for backend_name, backend_module in backends.items():
                self.backends[backend_name] = backend_module
                logger.info(f"Initialized backend: {backend_name}")
            
            # Load available models
            self._load_models()
            
            # Set default model
            if self.models:
                self.default_model = list(self.models.keys())[0]
                logger.info(f"Set default model: {self.default_model}")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing backends: {e}")
            return False
    
    def _discover_backends(self) -> Dict[str, Any]:
        """Discover available backends."""
        backends = {}
        
        # In a real implementation, this would dynamically discover backends
        # For now, we'll return a mock backend
        backends["mock"] = {
            "name": "mock",
            "query": lambda text, **kwargs: f"Mock response to: {text}"
        }
        
        return backends
    
    def _load_models(self):
        """Load available models from all backends."""
        for backend_name, backend in self.backends.items():
            try:
                # In a real implementation, this would query the backend for available models
                # For now, we'll add mock models
                if backend_name == "mock":
                    self.models["mock-7b"] = {
                        "backend": backend_name,
                        "name": "mock-7b",
                        "parameters": "7B",
                        "context_length": 8192
                    }
                    self.models["mock-13b"] = {
                        "backend": backend_name,
                        "name": "mock-13b",
                        "parameters": "13B",
                        "context_length": 8192
                    }
            except Exception as e:
                logger.error(f"Error loading models from backend {backend_name}: {e}")
    
    def map_role_to_model(self, role: str, model_name: str):
        """Map a role to a specific model."""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found, cannot map role {role}")
            return False
        
        self.role_mappings[role] = model_name
        logger.info(f"Mapped role {role} to model {model_name}")
        return True
    
    def query(self, text: str, role: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """Query a model based on role or explicit model name."""
        # Determine which model to use
        model_name = None
        
        if model:
            # Explicit model name provided
            model_name = model
        elif role and role in self.role_mappings:
            # Role mapping exists
            model_name = self.role_mappings[role]
        else:
            # Use default model
            model_name = self.default_model
        
        if not model_name or model_name not in self.models:
            logger.warning(f"Model {model_name} not found, using default model")
            model_name = self.default_model
        
        if not model_name:
            logger.error("No model available for query")
            return "Error: No model available for query"
        
        # Get model info
        model_info = self.models[model_name]
        backend_name = model_info["backend"]
        
        if backend_name not in self.backends:
            logger.error(f"Backend {backend_name} not found")
            return f"Error: Backend {backend_name} not found"
        
        # Query the model
        try:
            backend = self.backends[backend_name]
            response = backend["query"](text, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Error querying model {model_name}: {e}")
            return f"Error querying model: {str(e)}"

# Create singleton instance
model_router = ModelRouter()

def initialize_backends():
    """Initialize all available backends."""
    return model_router.initialize_backends()

def map_role_to_model(role: str, model_name: str):
    """Map a role to a specific model."""
    return model_router.map_role_to_model(role, model_name)

def query(text: str, role: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
    """Query a model based on role or explicit model name."""
    return model_router.query(text, role, model, **kwargs)

# Initialize on import if needed
if __name__ == "__main__":
    initialize_backends()
    map_role_to_model("consciousness", "mock-13b")
    response = query("Hello, world!", role="consciousness")
    print(response)