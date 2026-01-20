#!/usr/bin/env python3
# Systems/engine/core/role_initializer.py

import os
import sys
import importlib
import logging
import argparse
from typing import Dict, Any, Optional

from .role_manager import RoleManager
from .cpu_pinning_manager import CPUPinningManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RoleInitializer")

class RoleInitializer:
    """
    Initializes a service with a specific role.
    Handles module loading, CPU pinning, and environment setup.
    """
    
    def __init__(self, ship_id: str = None):
        """
        Initialize the role initializer.
        
        Args:
            ship_id: Ship identifier (defaults to environment variable)
        """
        self.ship_id = ship_id or os.environ.get("SHIP_ID", "unknown")
        self.role_manager = RoleManager(ship_id=self.ship_id)
        self.cpu_pinning_manager = CPUPinningManager()
        
        logger.info(f"Role Initializer created for ship {self.ship_id}")
    
    def initialize_role(self, role_name: str) -> Any:
        """
        Initialize a service with a specific role.
        
        Args:
            role_name: Name of the role to initialize
            
        Returns:
            Initialized role instance or None if failed
        """
        # Check if role exists
        if role_name not in self.role_manager.get_available_roles():
            logger.error(f"Unknown role: {role_name}")
            return None
        
        try:
            # Switch to the specified role
            if not self.role_manager.switch_role(role_name):
                logger.error(f"Failed to switch to role: {role_name}")
                return None
            
            # Get role definition
            role_def = self.role_manager.get_role_definition(role_name)
            
            # Apply CPU pinning
            self.cpu_pinning_manager.pin_current_process_to_role(role_name)
            
            # Import the module
            module_path = role_def.get("module_path")
            class_name = role_def.get("class_name")
            
            if not module_path or not class_name:
                logger.error(f"Invalid role definition for {role_name}: missing module_path or class_name")
                return None
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            role_class = getattr(module, class_name)
            
            # Initialize the role
            role_instance = role_class()
            
            logger.info(f"Successfully initialized role: {role_name}")
            return role_instance
        except Exception as e:
            logger.error(f"Error initializing role {role_name}: {str(e)}")
            return None
    
    def run_role(self, role_name: str) -> int:
        """
        Run a service with a specific role.
        
        Args:
            role_name: Name of the role to run
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Initialize the role
        role_instance = self.initialize_role(role_name)
        
        if not role_instance:
            logger.error(f"Failed to initialize role: {role_name}")
            return 1
        
        try:
            # Check if the role instance has a run method
            if hasattr(role_instance, "run") and callable(getattr(role_instance, "run")):
                # Run the role
                logger.info(f"Running role: {role_name}")
                role_instance.run()
                return 0
            else:
                logger.error(f"Role {role_name} does not have a run method")
                return 1
        except Exception as e:
            logger.error(f"Error running role {role_name}: {str(e)}")
            return 1

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Initialize and run a service role")
    parser.add_argument("role", help="Role name to initialize")
    parser.add_argument("--ship-id", help="Ship identifier")
    args = parser.parse_args()
    
    initializer = RoleInitializer(ship_id=args.ship_id)
    return initializer.run_role(args.role)

if __name__ == "__main__":
    sys.exit(main())
