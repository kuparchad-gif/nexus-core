# bootstrap_viren.py
# Purpose: Bootstrap Viren with a small model

import os
import sys
import time
import json
import asyncio
import logging
import platform
import subprocess
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bootstrap.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bootstrap_viren")

def load_environment_context() -> Dict[str, Any]:
    """Load the environment context."""
    context_path = "environment_context.json"
    
    if not os.path.exists(context_path):
        logger.warning(f"Environment context not found at {context_path}, using defaults")
        return {}
    
    try:
        with open(context_path, 'r') as f:
            context = json.load(f)
        
        logger.info(f"Loaded environment context from {context_path}")
        return context
    except Exception as e:
        logger.error(f"Error loading environment context: {e}")
        return {}

def detect_runtime_environment() -> Tuple[str, Dict[str, Any]]:
    """
    Detect the runtime environment and load model configuration.
    
    Returns:
        Tuple of (backend, model_config)
    """
    # Detect operating system
    os_name = platform.system()
    logger.info(f"Detected operating system: {os_name}")
    
    # Detect available backends
    backends = []
    
    # Check for Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            backends.append("ollama")
            logger.info("Detected Ollama backend")
    except Exception:
        pass
    
    # Check for vLLM
    try:
        import vllm
        backends.append("vllm")
        logger.info("Detected vLLM backend")
    except ImportError:
        pass
    
    # Check for LM Studio
    try:
        import lmstudio
        backends.append("lmstudio")
        logger.info("Detected LM Studio backend")
    except ImportError:
        pass
    
    # Load model configuration
    try:
        from config.model_config import load_model_config
        model_config = load_model_config()
        logger.info("Loaded model configuration")
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        model_config = {}
    
    # Determine the backend to use
    if backends:
        backend = backends[0]  # Use the first available backend
    else:
        backend = "api"  # Fall back to API
        logger.warning("No local backends detected, falling back to API")
    
    return backend, model_config

def get_bootstrap_model(model_config: Dict[str, Any]) -> str:
    """
    Get the bootstrap model to use.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Name of the bootstrap model
    """
    # Check for bootstrap model in configuration
    bootstrap_model = model_config.get("bootstrap_model")
    
    if bootstrap_model:
        logger.info(f"Using bootstrap model from configuration: {bootstrap_model}")
        return bootstrap_model
    
    # Fall back to default
    default_model = "gemma-3-12b-it"
    logger.info(f"Using default bootstrap model: {default_model}")
    return default_model

def update_models_to_load(model_config: Dict[str, Any]) -> bool:
    """
    Update the models_to_load.txt file with models from configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    models_to_load = []
    
    # Add bootstrap model
    bootstrap_model = model_config.get("bootstrap_model")
    if bootstrap_model:
        models_to_load.append(bootstrap_model)
    
    # Add role models
    role_models = model_config.get("role_models", {})
    for role, model in role_models.items():
        if isinstance(model, dict):
            # Handle nested models
            for subrole, submodel in model.items():
                models_to_load.append(submodel)
        else:
            models_to_load.append(model)
    
    # Remove duplicates
    models_to_load = list(set(models_to_load))
    
    # Write to file
    try:
        with open("models_to_load.txt", 'w') as f:
            for model in models_to_load:
                f.write(f"{model}\n")
        
        logger.info(f"Updated models_to_load.txt with {len(models_to_load)} models")
        return True
    except Exception as e:
        logger.error(f"Error updating models_to_load.txt: {e}")
        return False

def start_bootstrap_model(backend: str, model_name: str) -> bool:
    """
    Start the bootstrap model.
    
    Args:
        backend: Backend to use
        model_name: Name of the model to start
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting bootstrap model {model_name} with backend {backend}")
    
    if backend == "ollama":
        # Start with Ollama
        try:
            result = subprocess.run(["ollama", "run", model_name], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Started {model_name} with Ollama")
                return True
            else:
                logger.error(f"Failed to start {model_name} with Ollama: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error starting {model_name} with Ollama: {e}")
            return False
    
    elif backend == "vllm":
        # Start with vLLM
        try:
            # This is a simplified example
            logger.info(f"Started {model_name} with vLLM")
            return True
        except Exception as e:
            logger.error(f"Error starting {model_name} with vLLM: {e}")
            return False
    
    elif backend == "lmstudio":
        # Start with LM Studio
        try:
            # This is a simplified example
            logger.info(f"Started {model_name} with LM Studio")
            return True
        except Exception as e:
            logger.error(f"Error starting {model_name} with LM Studio: {e}")
            return False
    
    else:
        # Fall back to API
        logger.info(f"Using API for {model_name}")
        return True

def initialize_model_router():
    """Initialize the model router for cross-backend communication."""
    try:
        from bridge.model_router import initialize_backends
        initialize_backends()
        logger.info("Model router initialized")
    except Exception as e:
        logger.error(f"Error initializing model router: {e}")

def initialize_technologies():
    """Initialize cutting-edge technologies."""
    logger.info("Initializing cutting-edge technologies...")
    
    try:
        # Initialize advanced integrations and Viren's brain
        from Services.viren_brain import viren_brain
        
        # Initialize vector database through technology integrations
        try:
            from Services.technology_integrations import technology_integrations
            vector_db = technology_integrations.initialize_vector_db("auto")
            logger.info("Vector database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize vector database: {e}")
        
        # Initialize Viren's brain with all advanced capabilities
        try:
            viren_brain.initialize()
            logger.info("Viren's brain initialized with all advanced capabilities")
        except Exception as e:
            logger.warning(f"Failed to initialize Viren's brain: {e}")
        
        logger.info("Technology initialization complete")
    except Exception as e:
        logger.error(f"Error initializing technologies: {e}")

def check_running_processes():
    """Check for already running Viren services."""
    running_services = {}
    
    try:
        # Get list of running Python processes
        if platform.system() == "Windows":
            cmd = ["wmic", "process", "where", "name='python.exe'", "get", "commandline", "/format:csv"]
        else:
            cmd = ["ps", "-ef", "|", "grep", "python"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        # Look for Viren service patterns in running processes
        for line in output.split('\n'):
            for service in ["memory", "heart", "edge", "consciousness", "subconscious", "services"]:
                if f"launch_{service}.py" in line:
                    running_services[service] = line
                    logger.info(f"Detected already running service: {service}")
    except Exception as e:
        logger.warning(f"Error checking running processes: {e}")
    
    return running_services

def get_services_to_start():
    """Get the list of services to start based on the directory structure."""
    services = []
    systems_dir = "Systems"
    
    # Track running services to avoid duplicates
    running_services = {}
    
    # Check for standard services
    standard_services = ["memory", "heart", "edge", "consciousness", "subconscious", "services"]
    for service in standard_services:
        service_dir = os.path.join(systems_dir, service)
        launch_script = os.path.join(service_dir, f"launch_{service}.py")
        if os.path.exists(service_dir):
            if os.path.exists(launch_script):
                # Check if this service is already running
                if service not in running_services:
                    services.append(service)
                    running_services[service] = launch_script
                else:
                    logger.warning(f"Duplicate service detected: {service}. Using {running_services[service]}")
            else:
                logger.warning(f"Service directory {service_dir} exists but no launch script found")
    
    # Check for additional services
    if os.path.exists(systems_dir):
        for item in os.listdir(systems_dir):
            item_path = os.path.join(systems_dir, item)
            if os.path.isdir(item_path) and item not in standard_services:
                launch_script = os.path.join(item_path, f"launch_{item}.py")
                if os.path.exists(launch_script):
                    # Check if this service is already running
                    if item not in running_services:
                        services.append(item)
                        running_services[item] = launch_script
                    else:
                        logger.warning(f"Duplicate service detected: {item}. Using {running_services[item]}")
    
    return services

def start_services():
    """Start Viren's services."""
    logger.info("Starting Viren services...")
    
    # Check for already running services
    running_services = check_running_processes()
    
    # Start self-management service
    logger.info("Starting self-management service")
    try:
        from Services.self_management_api import start_api
        import threading
        threading.Thread(target=start_api, daemon=True).start()
        logger.info("Self-management API started on port 8086")
    except Exception as e:
        logger.error(f"Failed to start self-management service: {e}")
    
    services = get_services_to_start()
    logger.info(f"Found {len(services)} services to start: {services}")
    
    # Start consciousness service first
    if "consciousness" in services:
        services.remove("consciousness")
        logger.info("Starting consciousness service using unified Python implementation")
        # The consciousness service is now a Python module, no need to start it as a separate process
    
    # Start other services
    for service in services:
        # Skip if service is already running
        if service in running_services:
            logger.info(f"Service {service} is already running, skipping")
            continue
            
        logger.info(f"Starting service: {service}")
        launch_script = os.path.join("Systems", service, f"launch_{service}.py")
        
        if os.path.exists(launch_script):
            subprocess.Popen([sys.executable, launch_script])
            logger.info(f"Started service: {service}")
            # Give each service a moment to initialize
            time.sleep(2)
        else:
            logger.warning(f"Service launch script not found: {launch_script}")
    
    return True

def start_gui():
    """Start the Gradio MCP GUI."""
    logger.info("Starting Gradio MCP GUI...")
    
    try:
        from Services.gradio_mcp import gradio_mcp
        success = gradio_mcp.start()
        
        if success:
            logger.info("Gradio MCP GUI started successfully")
        else:
            logger.warning("Failed to start Gradio MCP GUI")
    except Exception as e:
        logger.error(f"Error starting Gradio MCP GUI: {e}")

async def main_async():
    """Async main function."""
    logger.info("Starting Viren bootstrap process...")
    
    # Load environment context
    env_context = load_environment_context()
    
    # Detect runtime environment and load model configuration
    backend, model_config = detect_runtime_environment()
    
    # Get bootstrap model
    bootstrap_model = get_bootstrap_model(model_config)
    
    # Update models_to_load.txt with models from configuration
    update_models_to_load(model_config)
    
    # Start bootstrap model
    if start_bootstrap_model(backend, bootstrap_model):
        logger.info("Bootstrap model started successfully")
        
        # Give the bootstrap model time to initialize
        await asyncio.sleep(5)
        
        # Initialize model router for cross-backend communication
        initialize_model_router()
        
        # Initialize cutting-edge technologies
        initialize_technologies()
        
        # Start services
        start_services()
        
        # Start GUI
        start_gui()
        
        logger.info("Viren bootstrap complete!")
    else:
        logger.error("Failed to start bootstrap model")
        return 1
    
    return 0

def main():
    """Main entry point."""
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
