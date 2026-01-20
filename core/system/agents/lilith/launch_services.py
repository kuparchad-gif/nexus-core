#!/usr/bin/env python
"""
Services Launcher
- Starts core services orchestration
- Connects to Heart and Memory services
- Launches LM Studio if available
- Initializes service nodes based on configuration
- Loads model configurations
"""

import os
import sys
import asyncio
import json
import logging
import time
import subprocess
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Import service node and model config
from Systems.service_core.service_node import ServiceNode
from Config.model_config import get_available_models, get_model_info, get_provider_info, get_deployment_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ServicesLauncher")

# Constants - Check all possible LM Studio locations
LM_STUDIO_PATHS = [
    "C:\\Program Files\\LM Studio\\LM Studio.exe",  # Admin install
    "C:\\Users\\Admin\\AppData\\Local\\Programs\\LM Studio\\LM Studio.exe",  # User install
    os.path.expanduser("~\\AppData\\Local\\Programs\\LM Studio\\LM Studio.exe"),  # Current user
    os.path.expanduser("~\\AppData\\Roaming\\LM Studio\\LM Studio.exe"),  # Roaming
    os.path.expanduser("~\\.lmstudio\\LM Studio.exe")  # .lmstudio folder
]
SERVICES_CONFIG_PATH = os.path.join(root_dir, "Config", "services_config.json")

class ServicesOrchestrator:
    """Orchestrates all services and their interactions"""
    
    def __init__(self):
        self.running = False
        self.services = {}
        self.service_nodes = {}
        self.lm_studio_process = None
        self.config = self._load_config()
        self.deployment_type = self._detect_deployment_type()
        self.available_models = {}
        self.active_model = None
    
    def _load_config(self):
        """Load services configuration"""
        default_config = {
            "launch_lm_studio": True,
            "heart_connection": True,
            "memory_connection": True,
            "preferred_provider": "Ollama",
            "services": {
                "text_processor": True,
                "tone_analyzer": True,
                "binary_processor": True,
                "service_node": True
            },
            "service_nodes": []
        }
        
        try:
            if os.path.exists(SERVICES_CONFIG_PATH):
                with open(SERVICES_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                return {**default_config, **config}
            else:
                logger.info(f"Config not found at {SERVICES_CONFIG_PATH}, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def _detect_deployment_type(self):
        """Detect deployment type based on environment"""
        # Check for cloud environment
        if os.environ.get("CLOUD_DEPLOYMENT") == "true":
            return "cloud"
        
        # Check for portable mode
        if os.path.exists(os.path.join(root_dir, "portable_mode")):
            return "portable"
        
        # Default to desktop
        return "desktop"
    
    def _load_model_configurations(self):
        """Load model configurations based on deployment type"""
        preferred_provider = self.config.get("preferred_provider", "Ollama")
        
        # Get available models for this deployment and provider
        models = get_available_models(self.deployment_type, preferred_provider)
        
        # Get model details
        for model_name in models:
            self.available_models[model_name] = get_model_info(model_name)
        
        # Select default model based on type
        if "phi:3-mini-4k" in self.available_models:
            self.active_model = "phi:3-mini-4k"
        elif "gemma:2b" in self.available_models:
            self.active_model = "gemma:2b"
        elif len(self.available_models) > 0:
            self.active_model = list(self.available_models.keys())[0]
        
        logger.info(f"Deployment type: {self.deployment_type}")
        logger.info(f"Available models: {list(self.available_models.keys())}")
        logger.info(f"Active model: {self.active_model}")
        
        # Get provider info
        provider_info = get_provider_info(preferred_provider)
        logger.info(f"Using provider: {preferred_provider} ({provider_info['type']})")
        
        # Get deployment info
        deployment_info = get_deployment_info(self.deployment_type)
        logger.info(f"Deployment resources: {deployment_info['resources']}")
    
    async def start(self):
        """Start the Services Orchestrator"""
        logger.info("Starting Services Orchestrator")
        
        self.running = True
        
        # Load model configurations
        self._load_model_configurations()
        
        # Launch LM Studio if configured
        if self.config["launch_lm_studio"]:
            self._launch_lm_studio()
        
        # Connect to Heart service
        if self.config["heart_connection"]:
            await self._connect_to_heart()
        
        # Connect to Memory service
        if self.config["memory_connection"]:
            await self._connect_to_memory()
        
        # Start individual services
        await self._start_services()
        
        # Initialize service nodes
        self._initialize_service_nodes()
        
        # Start monitoring task
        asyncio.create_task(self._monitor_services())
        
        logger.info("Services Orchestrator started")
        return True
    
    def _launch_lm_studio(self):
        """Launch LM Studio if available"""
        try:
            lm_studio_path = None
            for path in LM_STUDIO_PATHS:
                if os.path.exists(path):
                    lm_studio_path = path
                    break
            
            if lm_studio_path:
                logger.info(f"Launching LM Studio from {lm_studio_path}")
                self.lm_studio_process = subprocess.Popen([lm_studio_path])
                logger.info("LM Studio launched")
            else:
                logger.warning(f"LM Studio not found in any of these locations: {LM_STUDIO_PATHS}")
        except Exception as e:
            logger.error(f"Error launching LM Studio: {e}")
    
    async def _connect_to_heart(self):
        """Connect to Heart service"""
        logger.info("Connecting to Heart service")
        try:
            # In a real implementation, would establish connection to Heart service
            # For now, just simulate connection
            await asyncio.sleep(0.5)
            self.services["heart"] = {
                "connected": True,
                "last_heartbeat": time.time()
            }
            logger.info("Connected to Heart service")
        except Exception as e:
            logger.error(f"Error connecting to Heart service: {e}")
    
    async def _connect_to_memory(self):
        """Connect to Memory service"""
        logger.info("Connecting to Memory service")
        try:
            # In a real implementation, would establish connection to Memory service
            # For now, just simulate connection
            await asyncio.sleep(0.5)
            self.services["memory"] = {
                "connected": True,
                "last_access": time.time()
            }
            logger.info("Connected to Memory service")
        except Exception as e:
            logger.error(f"Error connecting to Memory service: {e}")
    
    async def _start_services(self):
        """Start individual services"""
        # Start text processor if configured
        if self.config["services"]["text_processor"]:
            await self._start_text_processor()
        
        # Start tone analyzer if configured
        if self.config["services"]["tone_analyzer"]:
            await self._start_tone_analyzer()
        
        # Start binary processor if configured
        if self.config["services"]["binary_processor"]:
            await self._start_binary_processor()
    
    def _initialize_service_nodes(self):
        """Initialize service nodes based on configuration"""
        if not self.config["services"].get("service_node", False):
            return
        
        for node_config in self.config.get("service_nodes", []):
            if not node_config.get("enabled", True):
                continue
            
            role = node_config.get("role")
            config_path = node_config.get("config_path")
            
            if not role or not config_path:
                continue
            
            try:
                # Create and initialize service node
                node = ServiceNode(role_config_path=os.path.join(root_dir, config_path))
                node.initialize()
                
                # Store node
                self.service_nodes[role] = node
                
                # Start pulse
                asyncio.create_task(self._node_pulse_loop(role))
                
                logger.info(f"Service node initialized: {role}")
            except Exception as e:
                logger.error(f"Error initializing service node {role}: {e}")
    
    async def _node_pulse_loop(self, role):
        """Run pulse loop for a service node"""
        node = self.service_nodes.get(role)
        if not node:
            return
        
        while self.running:
            try:
                node.pulse()
            except Exception as e:
                logger.error(f"Error in service node pulse {role}: {e}")
            
            await asyncio.sleep(node.pulse_interval)
    
    async def _start_text_processor(self):
        """Start text processor service"""
        logger.info("Starting text processor service")
        try:
            # In a real implementation, would start the actual service
            # For now, just simulate starting
            await asyncio.sleep(0.5)
            self.services["text_processor"] = {
                "running": True,
                "started_at": time.time(),
                "model": self.active_model
            }
            logger.info(f"Text processor service started with model {self.active_model}")
        except Exception as e:
            logger.error(f"Error starting text processor service: {e}")
    
    async def _start_tone_analyzer(self):
        """Start tone analyzer service"""
        logger.info("Starting tone analyzer service")
        try:
            # In a real implementation, would start the actual service
            # For now, just simulate starting
            await asyncio.sleep(0.5)
            self.services["tone_analyzer"] = {
                "running": True,
                "started_at": time.time()
            }
            logger.info("Tone analyzer service started")
        except Exception as e:
            logger.error(f"Error starting tone analyzer service: {e}")
    
    async def _start_binary_processor(self):
        """Start binary processor service"""
        logger.info("Starting binary processor service")
        try:
            # In a real implementation, would start the actual service
            # For now, just simulate starting
            await asyncio.sleep(0.5)
            self.services["binary_processor"] = {
                "running": True,
                "started_at": time.time()
            }
            logger.info("Binary processor service started")
        except Exception as e:
            logger.error(f"Error starting binary processor service: {e}")
    
    async def _monitor_services(self):
        """Monitor services and restart if needed"""
        while self.running:
            try:
                # Check LM Studio
                if self.lm_studio_process and self.lm_studio_process.poll() is not None:
                    logger.warning("LM Studio has stopped, restarting")
                    self._launch_lm_studio()
                
                # Check Heart connection
                if "heart" in self.services and self.services["heart"]["connected"]:
                    # In a real implementation, would check if connection is still active
                    # For now, just update last heartbeat
                    self.services["heart"]["last_heartbeat"] = time.time()
                
                # Check Memory connection
                if "memory" in self.services and self.services["memory"]["connected"]:
                    # In a real implementation, would check if connection is still active
                    # For now, just update last access
                    self.services["memory"]["last_access"] = time.time()
                
                # Check individual services
                for service_name in ["text_processor", "tone_analyzer", "binary_processor"]:
                    if service_name in self.services and self.services[service_name]["running"]:
                        # In a real implementation, would check if service is still running
                        # For now, just assume it's running
                        pass
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
            
            # Wait before next check
            await asyncio.sleep(10)
    
    def stop(self):
        """Stop the Services Orchestrator"""
        logger.info("Stopping Services Orchestrator")
        
        self.running = False
        
        # Stop LM Studio if it was launched
        if self.lm_studio_process:
            try:
                self.lm_studio_process.terminate()
                logger.info("LM Studio terminated")
            except Exception as e:
                logger.error(f"Error terminating LM Studio: {e}")
        
        # Stop individual services
        for service_name in ["text_processor", "tone_analyzer", "binary_processor"]:
            if service_name in self.services and self.services[service_name]["running"]:
                logger.info(f"Stopping {service_name} service")
                self.services[service_name]["running"] = False
        
        logger.info("Services Orchestrator stopped")

async def main():
    """Main entry point for Services"""
    logger.info("Starting Services...")
    
    try:
        # Initialize Services Orchestrator
        orchestrator = ServicesOrchestrator()
        await orchestrator.start()
        
        # Keep the service running
        while True:
            await asyncio.sleep(3600)  # 1 hour
            
    except Exception as e:
        logger.error(f"Error in Services: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())