#!/usr/bin/env python3
"""
System Integrator for Viren
Manages integration between different Viren instances and components
"""

import os
import json
import time
import logging
import importlib
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemIntegrator")

class SystemIntegrator:
    """Manages integration between different Viren instances and components"""
    
    def __init__(self, database_registry=None, binary_protocol=None):
        """Initialize the system integrator"""
        self.database_registry = database_registry
        self.binary_protocol = binary_protocol
        self.config_path = os.path.join('C:/Viren/config', 'system_integrator.json')
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load integrator configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default configuration
                self.config = {
                    "sync_interval": 3600,  # 1 hour
                    "last_sync": 0,
                    "auto_sync": True,
                    "sync_targets": ["weights", "capabilities", "databases"],
                    "remote_systems": [],
                    "soulseed": {
                        "enabled": True,
                        "path": "C:/Viren/config/viren_soulprint.json",
                        "sync_enabled": True
                    }
                }
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading integrator configuration: {e}")
            self.config = {}
    
    def _save_config(self) -> None:
        """Save integrator configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving integrator configuration: {e}")
    
    def initialize(self) -> bool:
        """Initialize the system integrator"""
        logger.info("Initializing System Integrator")
        
        # Check for Soulseed
        if self.config.get("soulseed", {}).get("enabled", True):
            soulseed_path = self.config.get("soulseed", {}).get("path", "C:/Viren/config/viren_soulprint.json")
            if os.path.exists(soulseed_path):
                logger.info(f"Soulseed found at {soulseed_path}")
                self._load_soulseed(soulseed_path)
            else:
                logger.warning(f"Soulseed not found at {soulseed_path}")
        
        # Perform initial sync if auto_sync is enabled
        if self.config.get("auto_sync", True):
            self.sync_with_remote_systems()
        
        return True
    
    def _load_soulseed(self, soulseed_path: str) -> bool:
        """Load Soulseed configuration"""
        try:
            with open(soulseed_path, 'r') as f:
                soulseed = json.load(f)
            
            logger.info(f"Loaded Soulseed: {soulseed.get('identity', {}).get('name', 'Unknown')}")
            
            # Register Soulseed with database registry
            if self.database_registry:
                self.database_registry.register_database(
                    "soulseed",
                    {
                        "name": "Viren Soulseed",
                        "type": "soulseed",
                        "description": "Core identity and configuration for Viren",
                        "data": soulseed
                    }
                )
            
            return True
        except Exception as e:
            logger.error(f"Error loading Soulseed: {e}")
            return False
    
    def sync_with_remote_systems(self) -> Dict[str, Any]:
        """Sync with remote Viren systems"""
        if not self.database_registry:
            logger.warning("Database registry not available, cannot sync")
            return {
                "success": False,
                "error": "Database registry not available"
            }
        
        try:
            # Get remote systems from config
            remote_systems = self.config.get("remote_systems", [])
            
            # If no remote systems specified, get from database registry
            if not remote_systems and self.database_registry:
                systems = self.database_registry.registry.get("systems", {})
                remote_systems = [
                    {
                        "id": system_id,
                        "name": system_info.get("name", f"Viren-{system_id[:8]}"),
                        "url": f"https://{system_info.get('name', 'viren')}.example.com/api"
                    }
                    for system_id, system_info in systems.items()
                    if system_id != self.database_registry.system_id
                ]
            
            synced_with = []
            errors = []
            
            for system in remote_systems:
                system_id = system.get("id")
                system_url = system.get("url")
                
                if not system_id or not system_url:
                    errors.append(f"Invalid system configuration: {system}")
                    continue
                
                logger.info(f"Syncing with remote system: {system.get('name', system_id)}")
                
                # In a real system, we would make API calls to the remote system
                # For now, we'll just simulate syncing
                
                # Sync databases if enabled
                if "databases" in self.config.get("sync_targets", []):
                    logger.info(f"Syncing databases with {system.get('name', system_id)}")
                    # Simulate receiving remote registry
                    remote_registry = {
                        "systems": {
                            system_id: {
                                "name": system.get("name", f"Viren-{system_id[:8]}"),
                                "type": "cloud" if "cloud" in system.get("name", "").lower() else "desktop",
                                "last_seen": time.time()
                            }
                        },
                        "databases": {},
                        "relationships": {}
                    }
                    
                    # Sync with remote registry
                    self.database_registry.sync_with_remote(remote_registry)
                
                # Sync weights if enabled
                if "weights" in self.config.get("sync_targets", []):
                    logger.info(f"Syncing weights with {system.get('name', system_id)}")
                    # In a real system, we would download weights from the remote system
                
                # Sync capabilities if enabled
                if "capabilities" in self.config.get("sync_targets", []):
                    logger.info(f"Syncing capabilities with {system.get('name', system_id)}")
                    # In a real system, we would exchange capability information
                
                synced_with.append(system_id)
            
            # Update last sync time
            self.config["last_sync"] = time.time()
            self._save_config()
            
            return {
                "success": True,
                "synced_with": synced_with,
                "errors": errors
            }
        
        except Exception as e:
            logger.error(f"Error syncing with remote systems: {e}")
            return {
                "success": False,
                "error": f"Error syncing with remote systems: {str(e)}"
            }
    
    def integrate_new_module(self, module_path: str) -> Dict[str, Any]:
        """Integrate a new module into the system"""
        try:
            # Check if module exists
            if not os.path.exists(module_path):
                return {
                    "success": False,
                    "error": f"Module not found at {module_path}"
                }
            
            # Get module name
            module_name = os.path.basename(module_path).replace(".py", "")
            
            # In a real system, we would analyze the module and integrate it
            # For now, we'll just simulate integration
            logger.info(f"Integrating new module: {module_name}")
            
            # Simulate module analysis
            module_type = "unknown"
            if "model" in module_name.lower():
                module_type = "model"
            elif "data" in module_name.lower():
                module_type = "data_processor"
            elif "api" in module_name.lower():
                module_type = "api"
            
            # Register with database registry
            if self.database_registry:
                self.database_registry.register_database(
                    f"module_{module_name}",
                    {
                        "name": f"Module: {module_name}",
                        "type": "module",
                        "module_type": module_type,
                        "path": module_path,
                        "integration_date": time.time()
                    }
                )
            
            return {
                "success": True,
                "module_name": module_name,
                "module_type": module_type,
                "path": module_path
            }
        
        except Exception as e:
            logger.error(f"Error integrating module: {e}")
            return {
                "success": False,
                "error": f"Error integrating module: {str(e)}"
            }
    
    def sync_soulseed(self) -> Dict[str, Any]:
        """Sync Soulseed with other Viren instances"""
        if not self.config.get("soulseed", {}).get("sync_enabled", True):
            return {
                "success": False,
                "error": "Soulseed sync is disabled"
            }
        
        soulseed_path = self.config.get("soulseed", {}).get("path", "C:/Viren/config/viren_soulprint.json")
        if not os.path.exists(soulseed_path):
            return {
                "success": False,
                "error": f"Soulseed not found at {soulseed_path}"
            }
        
        try:
            # In a real system, we would sync the Soulseed with other instances
            # For now, we'll just simulate syncing
            logger.info("Syncing Soulseed with other Viren instances")
            
            # Get remote systems
            remote_systems = self.config.get("remote_systems", [])
            
            synced_with = []
            errors = []
            
            for system in remote_systems:
                system_id = system.get("id")
                system_name = system.get("name", system_id)
                
                logger.info(f"Syncing Soulseed with {system_name}")
                synced_with.append(system_id)
            
            return {
                "success": True,
                "synced_with": synced_with,
                "errors": errors
            }
        
        except Exception as e:
            logger.error(f"Error syncing Soulseed: {e}")
            return {
                "success": False,
                "error": f"Error syncing Soulseed: {str(e)}"
            }
    
    def add_remote_system(self, system_id: str, system_name: str, system_url: str) -> bool:
        """Add a remote system for syncing"""
        remote_systems = self.config.get("remote_systems", [])
        
        # Check if system already exists
        for system in remote_systems:
            if system.get("id") == system_id:
                system["name"] = system_name
                system["url"] = system_url
                self._save_config()
                return True
        
        # Add new system
        remote_systems.append({
            "id": system_id,
            "name": system_name,
            "url": system_url
        })
        
        self.config["remote_systems"] = remote_systems
        return self._save_config()
    
    def get_remote_systems(self) -> List[Dict[str, Any]]:
        """Get list of remote systems"""
        return self.config.get("remote_systems", [])