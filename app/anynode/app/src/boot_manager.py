#!/usr/bin/env python3
"""
Boot Manager for Viren
Ensures reliable system initialization with dependency checks
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BootManager")

class BootManager:
    """Manages the boot sequence for Viren"""
    
    def __init__(self, config_path: str = None):
        """Initialize the boot manager"""
        self.config_path = config_path or os.path.join('C:/Viren/config', 'boot_config.json')
        self.components = {}
        self.dependencies = {}
        self.boot_order = []
        self.status = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load boot configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.components = config.get('components', {})
                    self.dependencies = config.get('dependencies', {})
            else:
                # Create default configuration
                self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading boot configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default boot configuration"""
        self.components = {
            "database_registry": {
                "module": "core.database_registry",
                "class": "DatabaseRegistry",
                "required": True
            },
            "binary_protocol": {
                "module": "core.binary_protocol",
                "class": "BinaryProtocol",
                "required": True
            },
            "weaviate": {
                "module": "vector.weaviate_client",
                "class": "WeaviateClient",
                "required": True
            },
            "intelligence_router": {
                "module": "core.intelligence_router",
                "class": "IntelligenceRouter",
                "required": True
            },
            "model_manager": {
                "module": "cloud.models",
                "class": "ModelManager",
                "required": True
            },
            "weight_manager": {
                "module": "core.weight_manager",
                "class": "WeightManager",
                "required": False
            }
        }
        
        self.dependencies = {
            "intelligence_router": ["database_registry", "binary_protocol", "weaviate"],
            "weight_manager": ["model_manager"]
        }
        
        self._save_config()
    
    def _save_config(self) -> None:
        """Save boot configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump({
                    'components': self.components,
                    'dependencies': self.dependencies
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving boot configuration: {e}")
    
    def _resolve_dependencies(self) -> bool:
        """Resolve component dependencies and create boot order"""
        visited = set()
        temp_visited = set()
        boot_order = []
        
        def visit(component):
            if component in temp_visited:
                logger.error(f"Circular dependency detected involving {component}")
                return False
            
            if component in visited:
                return True
            
            temp_visited.add(component)
            
            # Visit dependencies first
            for dep in self.dependencies.get(component, []):
                if dep not in self.components:
                    logger.error(f"Missing dependency: {dep} for component {component}")
                    return False
                
                if not visit(dep):
                    return False
            
            temp_visited.remove(component)
            visited.add(component)
            boot_order.append(component)
            return True
        
        # Visit all components
        for component in self.components:
            if component not in visited:
                if not visit(component):
                    return False
        
        self.boot_order = boot_order
        return True
    
    def _import_component(self, component_name: str) -> Tuple[bool, Any]:
        """Import a component module and class"""
        try:
            component = self.components[component_name]
            module_name = component["module"]
            class_name = component["class"]
            
            # Import module
            module = __import__(module_name, fromlist=[class_name])
            
            # Get class
            component_class = getattr(module, class_name)
            
            return True, component_class
        except ImportError as e:
            logger.error(f"Failed to import module {component['module']}: {e}")
            return False, None
        except AttributeError as e:
            logger.error(f"Failed to find class {component['class']}: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Error importing component {component_name}: {e}")
            return False, None
    
    def boot(self) -> bool:
        """Boot the system in the correct dependency order"""
        logger.info("Starting Viren boot sequence")
        
        # Resolve dependencies
        if not self._resolve_dependencies():
            logger.error("Failed to resolve dependencies")
            return False
        
        logger.info(f"Boot order: {self.boot_order}")
        
        # Initialize components
        instances = {}
        
        for component_name in self.boot_order:
            component = self.components[component_name]
            required = component.get("required", True)
            
            logger.info(f"Booting component: {component_name}")
            
            # Import component
            success, component_class = self._import_component(component_name)
            if not success:
                if required:
                    logger.error(f"Failed to import required component: {component_name}")
                    return False
                else:
                    logger.warning(f"Failed to import optional component: {component_name}")
                    self.status[component_name] = "failed"
                    continue
            
            # Initialize component
            try:
                # Get dependencies
                deps = {}
                for dep in self.dependencies.get(component_name, []):
                    if dep in instances:
                        deps[dep] = instances[dep]
                
                # Initialize with dependencies if needed
                if deps and hasattr(component_class, "__init__") and component_class.__init__.__code__.co_argcount > 1:
                    instance = component_class(**deps)
                else:
                    instance = component_class()
                
                # Initialize if method exists
                if hasattr(instance, "initialize"):
                    success = instance.initialize()
                    if not success and required:
                        logger.error(f"Failed to initialize required component: {component_name}")
                        return False
                
                instances[component_name] = instance
                self.status[component_name] = "running"
                logger.info(f"Successfully booted component: {component_name}")
            
            except Exception as e:
                logger.error(f"Error initializing component {component_name}: {e}")
                if required:
                    return False
                self.status[component_name] = "failed"
        
        logger.info("Viren boot sequence completed successfully")
        return True
    
    def get_status(self) -> Dict[str, str]:
        """Get the status of all components"""
        return self.status
    
    def verify_boot(self) -> Tuple[bool, Dict[str, str]]:
        """Verify that all required components are running"""
        all_required_running = True
        
        for component_name, component in self.components.items():
            if component.get("required", True) and self.status.get(component_name) != "running":
                all_required_running = False
                break
        
        return all_required_running, self.status