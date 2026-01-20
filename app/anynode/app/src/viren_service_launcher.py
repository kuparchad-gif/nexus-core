# Systems/services/viren_service_launcher.py
# Purpose: Launch Viren's services

import os
import sys
import logging
import importlib
import threading
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger("viren_services")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define services
SERVICES = [
    {
        "name": "model_service",
        "module": "Systems.services.model_service",
        "function": "start_service",
        "required": True,
        "args": [],
        "kwargs": {}
    },
    {
        "name": "memory_service",
        "module": "Systems.memory.memory_initializer",
        "function": "initialize",
        "required": True,
        "args": [],
        "kwargs": {}
    },
    {
        "name": "bridge_service",
        "module": "bridge.bridge_engine",
        "function": "start_bridge",
        "required": True,
        "args": [],
        "kwargs": {}
    },
    {
        "name": "sentinel_service",
        "module": "sentinel_mode.sentinel_start",
        "function": "start_sentinel",
        "required": False,
        "args": [],
        "kwargs": {}
    }
]

def import_module_function(module_path: str, function_name: str):
    """Import a function from a module."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except ImportError:
        logger.error(f"Failed to import module: {module_path}")
        return None
    except AttributeError:
        logger.error(f"Function {function_name} not found in module {module_path}")
        return None

def launch_service(service: Dict[str, Any]):
    """Launch a service."""
    name = service["name"]
    module_path = service["module"]
    function_name = service["function"]
    required = service["required"]
    args = service["args"]
    kwargs = service["kwargs"]
    
    logger.info(f"Launching service: {name}")
    
    try:
        # Import the function
        func = import_module_function(module_path, function_name)
        if func is None:
            if required:
                logger.error(f"Required service {name} could not be launched")
                return False
            else:
                logger.warning(f"Optional service {name} could not be launched")
                return True
        
        # Call the function
        result = func(*args, **kwargs)
        
        logger.info(f"Service {name} launched successfully")
        return True
    except Exception as e:
        if required:
            logger.error(f"Error launching required service {name}: {e}")
            return False
        else:
            logger.warning(f"Error launching optional service {name}: {e}")
            return True

def launch_all():
    """Launch all services."""
    logger.info("Launching all services...")
    
    # Launch required services first
    required_services = [s for s in SERVICES if s["required"]]
    optional_services = [s for s in SERVICES if not s["required"]]
    
    # Launch required services
    for service in required_services:
        success = launch_service(service)
        if not success:
            logger.error(f"Failed to launch required service: {service['name']}")
            return False
    
    # Launch optional services in threads
    threads = []
    for service in optional_services:
        thread = threading.Thread(target=launch_service, args=(service,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for optional services to start
    for thread in threads:
        thread.join(timeout=5)
    
    logger.info("All services launched")
    return True

def launch_model_service():
    """Launch the model service."""
    # Import model configuration
    try:
        from config.model_config import load_model_config, initialize_model_loader
        
        # Initialize model loader
        config = initialize_model_loader()
        
        # Start model service
        import ollama
        
        # Check if Gemma 2B is available
        models = ollama.list()
        gemma_available = any(model["name"].startswith("gemma:2b") for model in models["models"])
        
        if not gemma_available:
            logger.info("Downloading boot model (Gemma 2B)...")
            ollama.pull("gemma:2b")
        
        logger.info("Model service started")
        return True
    except Exception as e:
        logger.error(f"Error starting model service: {e}")
        return False

# Add model service to services
SERVICES.append({
    "name": "model_service_init",
    "module": __name__,
    "function": "launch_model_service",
    "required": True,
    "args": [],
    "kwargs": {}
})

if __name__ == "__main__":
    launch_all()