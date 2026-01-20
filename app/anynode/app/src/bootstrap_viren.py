#!/usr/bin/env python3
"""
Bootstrap Viren - System initialization and startup
"""

import os
import sys
import json
import time
import logging
import subprocess
import importlib.util
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='viren_bootstrap.log'
),
logger = logging.getLogger("VirenBootstrap")

class VirenBootstrap:
    """
    Bootstrap process for Viren AI system.
    """
    
    def __init__(self, deployment_type: str = "desktop"):
        """
        Initialize the bootstrap process.
        
        Args:
            deployment_type: Type of deployment (desktop, portable, cloud)
        """
        self.deployment_type = deployment_type
        self.config = {}
        self.services = {}
        self.start_time = time.time()
        
        # Ensure required directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            "config",
            "logs",
            "data",
            "public",
            "models"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_config(self) -> bool:
        """
        Load configuration from files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load soulprint
            soulprint_path = os.path.join("config", "viren_soulprint.json")
            if os.path.exists(soulprint_path):
                with open(soulprint_path, 'r') as f:
                    self.config["soulprint"] = json.load(f)
                logger.info("Loaded soulprint configuration")
            else:
                logger.warning("Soulprint configuration not found")
                return False
            
            # Load model configuration
            try:
                spec = importlib.util.spec_from_file_location(
                    "model_config", 
                    os.path.join("config", "model_config.py")
                )
                model_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_config)
                
                self.config["models"] = model_config.MODELS
                self.config["providers"] = model_config.PROVIDERS
                self.config["deployments"] = model_config.DEPLOYMENTS
                logger.info("Loaded model configuration")
            except Exception as e:
                logger.error(f"Error loading model configuration: {e}")
                return False
            
            # Set deployment-specific configuration
            if self.deployment_type in self.config["deployments"]:
                self.config["deployment"] = self.config["deployments"][self.deployment_type]
                logger.info(f"Using {self.deployment_type} deployment configuration")
            else:
                logger.warning(f"Deployment type {self.deployment_type} not found, using desktop")
                self.config["deployment"] = self.config["deployments"]["desktop"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are installed.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        required_packages = [
            "gradio",
            "matplotlib",
            "networkx",
            "requests",
            "psutil",
            "pyttsx3",
            "SpeechRecognition"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing dependencies: {', '.join(missing_packages)}")
            logger.info("Attempting to install missing dependencies...")
            
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", *missing_packages
                ])
                logger.info("Successfully installed missing dependencies")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                return False
        
        return True
    
    def check_model_provider(self) -> bool:
        """
        Check if the model provider is available.
        
        Returns:
            True if available, False otherwise
        """
        provider = self.config["soulprint"]["preferences"]["default_provider"]
        
        # Check for cloud models first
        try:
            import modal
            logger.info("Checking for cloud models...")
            try:
                modal.Function.lookup("aethereal-nexus", "tiny_llama")
                logger.info("Cloud models available")
                return True
            except Exception as e:
                logger.warning(f"Cloud models not available: {e}")
                # Continue with local provider check
        except ImportError:
            logger.warning("Modal package not installed, skipping cloud model check")
        
        # Check local providers
        if provider == "Ollama":
            return self._check_ollama()
        elif provider == "vLLM":
            return self._check_vllm()
        elif provider == "API":
            return True  # Assume API is always available
        else:
            logger.error(f"Unknown provider: {provider}")
            return False
    
    def _check_ollama(self) -> bool:
        """
        Check if Ollama is installed and running.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Check if Ollama is installed
            result = subprocess.run(
                ["ollama", "version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("Ollama is not installed or not in PATH")
                return False
            
            # Check if Ollama server is running
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            
            if response.status_code != 200:
                logger.error("Ollama server is not running")
                return False
            
            logger.info("Ollama is installed and running")
            return True
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            return False
    
    def _check_vllm(self) -> bool:
        """
        Check if vLLM is installed and running.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Check if vLLM is installed
            import importlib.util
            if importlib.util.find_spec("vllm") is None:
                logger.error("vLLM is not installed")
                return False
            
            # Check if vLLM server is running
            import requests
            response = requests.get("http://localhost:8000/v1/models")
            
            if response.status_code != 200:
                logger.error("vLLM server is not running")
                return False
            
            logger.info("vLLM is installed and running")
            return True
        except Exception as e:
            logger.error(f"Error checking vLLM: {e}")
            return False
    
    def initialize_services(self) -> bool:
        """
        Initialize required services.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import required modules
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Initialize model service
            from model_service import ModelService
            self.services["model"] = ModelService()
            
            # Set active model
            default_model = self.config["soulprint"]["preferences"]["default_model"]
            default_provider = self.config["soulprint"]["preferences"]["default_provider"]
            
            if not self.services["model"].set_active_model(default_model, default_provider):
                logger.warning(f"Failed to set active model {default_model} with provider {default_provider}")
                
                # Try to find an available model
                available_models = self.services["model"].get_available_models(provider=default_provider)
                if available_models:
                    logger.info(f"Using alternative model: {available_models[0]}")
                    self.services["model"].set_active_model(available_models[0], default_provider)
            
            # Initialize Gray's Anatomy
            from grays_anatomy import GraysAnatomy
            self.services["anatomy"] = GraysAnatomy("viren")
            
            logger.info("Services initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            return False
    
    def start_mcp(self) -> bool:
        """
        Start the Mission Control Panel.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting Mission Control Panel...")
            
            # Check if viren_mcp.py exists
            if not os.path.exists("viren_mcp.py"):
                logger.error("viren_mcp.py not found")
                return False
            
            # Start MCP in a new process
            logger.info("Launching MCP with cloud model support...")
            subprocess.Popen([sys.executable, "viren_mcp.py"])
            
            logger.info("Mission Control Panel started")
            return True
        except Exception as e:
            logger.error(f"Error starting Mission Control Panel: {e}")
            return False
    
    def bootstrap(self) -> bool:
        """
        Run the bootstrap process.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting Viren bootstrap process for {self.deployment_type} deployment")
        
        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration...")
        if not self.load_config():
            logger.error("Failed to load configuration")
            return False
        
        # Step 2: Check dependencies
        logger.info("Step 2: Checking dependencies...")
        if not self.check_dependencies():
            logger.error("Failed to check dependencies")
            return False
        
        # Step 3: Check model provider
        logger.info("Step 3: Checking model provider...")
        if not self.check_model_provider():
            logger.error("Failed to check model provider")
            return False
        
        # Step 4: Initialize services
        logger.info("Step 4: Initializing services...")
        if not self.initialize_services():
            logger.error("Failed to initialize services")
            return False
        
        # Step 5: Start MCP
        logger.info("Step 5: Starting Mission Control Panel...")
        if not self.start_mcp():
            logger.error("Failed to start Mission Control Panel")
            return False
        
        # Bootstrap complete
        elapsed_time = time.time() - self.start_time
        logger.info(f"Bootstrap complete in {elapsed_time:.2f} seconds")
        return True

# Main function
def main():
    # Get deployment type from command line arguments
    deployment_type = "desktop"
    if len(sys.argv) > 1:
        deployment_type = sys.argv[1]
    
    # Create bootstrap instance
    bootstrap = VirenBootstrap(deployment_type)
    
    # Run bootstrap process
    success = bootstrap.bootstrap()
    
    if not success:
        print("Bootstrap failed. Check viren_bootstrap.log for details.")
        sys.exit(1)
    
    print("Viren bootstrap complete. Mission Control Panel should be starting...")

if __name__ == "__main__":
    main()
