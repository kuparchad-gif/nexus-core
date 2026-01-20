# Systems/Config/viren_runtime.py
# Purpose: Runtime configuration for Viren

import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("viren_runtime")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/viren_runtime.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class VirenRuntime:
    """
    Runtime configuration for Viren.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the runtime configuration."""
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "runtime_config.json")
        self.config = self._load_config()
        
        logger.info("Viren runtime configuration initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the runtime configuration."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded runtime configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default runtime configuration."""
        return {
            "version": "1.0.0",
            "environment": "development",
            "services": {
                "memory": {
                    "enabled": True,
                    "path": "memory"
                },
                "consciousness": {
                    "enabled": True,
                    "model": "default"
                },
                "heart": {
                    "enabled": True,
                    "pulse_rate": 60
                }
            },
            "advanced": {
                "lora": {
                    "enabled": True,
                    "path": "models/lora"
                },
                "diffusers": {
                    "enabled": True,
                    "path": "models/diffusers"
                },
                "monitoring": {
                    "enabled": True,
                    "port": 8000
                }
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current runtime configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update the runtime configuration.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        # Update the configuration
        self._update_dict(self.config, updates)
        
        # Save the updated configuration
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Updated runtime configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _update_dict(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            target: Dictionary to update
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value

# Create a singleton instance
viren_runtime = VirenRuntime()
