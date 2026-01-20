#!/usr/bin/env python3
"""
Intelligence Router for Viren
Routes complex tasks to appropriate modules and orchestrates execution
"""

import os
import re
import json
import time
import logging
import importlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntelligenceRouter")

class IntelligenceRouter:
    """Routes complex tasks to appropriate modules and orchestrates execution"""
    
    def __init__(self, database_registry=None, binary_protocol=None, weaviate=None):
        """Initialize the intelligence router"""
        self.database_registry = database_registry
        self.binary_protocol = binary_protocol
        self.weaviate = weaviate
        self.capabilities = {}
        self.modules = {}
        self.config_path = os.path.join('C:/Viren/config', 'intelligence_router.json')
        self._load_config()
    
    def _load_config(self) -> None:
        """Load router configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.capabilities = config.get('capabilities', {})
            else:
                # Create default configuration
                self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading router configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default router configuration"""
        self.capabilities = {
            "2d_to_3d_conversion": {
                "module": "capabilities.image_processing",
                "class": "Image2DTo3DConverter",
                "description": "Converts 2D images to 3D models",
                "inputs": ["image_2d"],
                "outputs": ["model_3d"],
                "requirements": ["torch", "opencv-python"]
            },
            "3d_model_texturing": {
                "module": "capabilities.model_processing",
                "class": "ModelTexturer",
                "description": "Applies textures to 3D models",
                "inputs": ["model_3d", "texture_description"],
                "outputs": ["textured_model_3d"],
                "requirements": ["blender-python", "pillow"]
            },
            "game_environment_creation": {
                "module": "capabilities.game_development",
                "class": "EnvironmentCreator",
                "description": "Creates game environments from 3D models",
                "inputs": ["textured_model_3d", "environment_description"],
                "outputs": ["game_environment"],
                "requirements": ["unity-python", "godot-python"]
            },
            "server_deployment": {
                "module": "capabilities.deployment",
                "class": "ServerDeployer",
                "description": "Deploys servers for multiplayer games",
                "inputs": ["game_environment", "server_requirements"],
                "outputs": ["deployed_server"],
                "requirements": ["docker", "kubernetes-python"]
            }
        }
        
        self._save_config()
    
    def _save_config(self) -> None:
        """Save router configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump({
                    'capabilities': self.capabilities
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving router configuration: {e}")
    
    def initialize(self) -> bool:
        """Initialize the intelligence router"""
        logger.info("Initializing Intelligence Router")
        
        # Register capabilities with the database registry
        if self.database_registry:
            try:
                self.database_registry.register_database(
                    "intelligence_router_capabilities",
                    {
                        "name": "Intelligence Router Capabilities",
                        "type": "capability_registry",
                        "description": "Registry of all capabilities available to the Intelligence Router",
                        "data": self.capabilities
                    }
                )
            except Exception as e:
                logger.error(f"Failed to register capabilities with database registry: {e}")
        
        return True
    
    def _parse_task(self, task_description: str) -> List[Dict[str, Any]]:
        """Parse a task description into subtasks"""
        # This is a simplified version - in a real system, this would use NLP
        subtasks = []
        
        # Check for 2D to 3D conversion
        if "2d" in task_description.lower() and "3d" in task_description.lower():
            subtasks.append({
                "capability": "2d_to_3d_conversion",
                "inputs": {"image_2d": "extract_from_context"},
                "description": "Convert 2D image to 3D model"
            })
        
        # Check for texturing
        if "texture" in task_description.lower() or "paint" in task_description.lower():
            subtasks.append({
                "capability": "3d_model_texturing",
                "inputs": {
                    "model_3d": "from_previous_step",
                    "texture_description": "extract_from_context"
                },
                "description": "Apply textures to 3D model"
            })
        
        # Check for game environment
        if "game" in task_description.lower() or "environment" in task_description.lower():
            subtasks.append({
                "capability": "game_environment_creation",
                "inputs": {
                    "textured_model_3d": "from_previous_step",
                    "environment_description": "extract_from_context"
                },
                "description": "Create game environment"
            })
        
        # Check for server deployment
        if "server" in task_description.lower() or "deploy" in task_description.lower():
            subtasks.append({
                "capability": "server_deployment",
                "inputs": {
                    "game_environment": "from_previous_step",
                    "server_requirements": "extract_from_context"
                },
                "description": "Deploy game server"
            })
        
        return subtasks
    
    def _load_capability_module(self, capability_name: str) -> Tuple[bool, Any]:
        """Load a capability module"""
        if capability_name not in self.capabilities:
            logger.error(f"Unknown capability: {capability_name}")
            return False, None
        
        capability = self.capabilities[capability_name]
        
        # Check if already loaded
        if capability_name in self.modules:
            return True, self.modules[capability_name]
        
        try:
            # Check requirements
            self._ensure_requirements(capability.get("requirements", []))
            
            # Import module
            module_name = capability["module"]
            class_name = capability["class"]
            
            module = importlib.import_module(module_name)
            capability_class = getattr(module, class_name)
            
            # Instantiate
            instance = capability_class()
            
            # Store in modules
            self.modules[capability_name] = instance
            
            return True, instance
        
        except ImportError as e:
            logger.error(f"Failed to import module {capability['module']}: {e}")
            return False, None
        except AttributeError as e:
            logger.error(f"Failed to find class {capability['class']}: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Error loading capability {capability_name}: {e}")
            return False, None
    
    def _ensure_requirements(self, requirements: List[str]) -> bool:
        """Ensure all requirements are installed"""
        import subprocess
        import sys
        
        for req in requirements:
            try:
                importlib.import_module(req.split("==")[0].replace("-", "_"))
            except ImportError:
                logger.info(f"Installing requirement: {req}")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                except Exception as e:
                    logger.error(f"Failed to install requirement {req}: {e}")
                    return False
        
        return True
    
    def _extract_from_context(self, input_name: str, context: Dict[str, Any]) -> Any:
        """Extract input from context"""
        # This is a simplified version - in a real system, this would be more sophisticated
        if input_name in context:
            return context[input_name]
        
        # Try to find in task description
        if "task_description" in context:
            # Very basic extraction - in reality would use NLP
            if input_name == "image_2d" and "image" in context:
                return context["image"]
            elif input_name == "texture_description" and "task_description" in context:
                return context["task_description"]
            elif input_name == "environment_description" and "task_description" in context:
                return context["task_description"]
            elif input_name == "server_requirements" and "task_description" in context:
                return {"type": "basic", "players": 4}  # Default
        
        return None
    
    def process_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a complex task"""
        logger.info(f"Processing task: {task_description}")
        
        context = context or {}
        context["task_description"] = task_description
        
        # Parse task into subtasks
        subtasks = self._parse_task(task_description)
        
        if not subtasks:
            return {
                "success": False,
                "error": "Could not parse task into subtasks",
                "task_description": task_description
            }
        
        # Execute subtasks
        results = {}
        
        for i, subtask in enumerate(subtasks):
            capability_name = subtask["capability"]
            logger.info(f"Executing subtask {i+1}/{len(subtasks)}: {subtask['description']}")
            
            # Load capability module
            success, module = self._load_capability_module(capability_name)
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to load capability: {capability_name}",
                    "completed_subtasks": i,
                    "results": results
                }
            
            # Prepare inputs
            inputs = {}
            for input_name, input_source in subtask["inputs"].items():
                if input_source == "from_previous_step" and i > 0:
                    # Get from previous subtask result
                    prev_capability = subtasks[i-1]["capability"]
                    if prev_capability in results:
                        inputs[input_name] = results[prev_capability]["outputs"].get(
                            self.capabilities[prev_capability]["outputs"][0]
                        )
                elif input_source == "extract_from_context":
                    inputs[input_name] = self._extract_from_context(input_name, context)
                else:
                    inputs[input_name] = input_source
            
            # Execute capability
            try:
                if hasattr(module, "execute"):
                    result = module.execute(**inputs)
                    results[capability_name] = {
                        "success": True,
                        "outputs": result
                    }
                else:
                    logger.error(f"Capability {capability_name} has no execute method")
                    return {
                        "success": False,
                        "error": f"Capability {capability_name} has no execute method",
                        "completed_subtasks": i,
                        "results": results
                    }
            except Exception as e:
                logger.error(f"Error executing capability {capability_name}: {e}")
                return {
                    "success": False,
                    "error": f"Error executing capability {capability_name}: {str(e)}",
                    "completed_subtasks": i,
                    "results": results
                }
        
        return {
            "success": True,
            "completed_subtasks": len(subtasks),
            "results": results
        }
    
    def register_capability(self, capability_name: str, capability_info: Dict[str, Any]) -> bool:
        """Register a new capability"""
        if capability_name in self.capabilities:
            logger.warning(f"Capability {capability_name} already exists, updating")
        
        self.capabilities[capability_name] = capability_info
        
        # Remove from loaded modules if exists
        if capability_name in self.modules:
            del self.modules[capability_name]
        
        # Update database registry
        if self.database_registry:
            try:
                self.database_registry.register_database(
                    "intelligence_router_capabilities",
                    {
                        "name": "Intelligence Router Capabilities",
                        "type": "capability_registry",
                        "description": "Registry of all capabilities available to the Intelligence Router",
                        "data": self.capabilities
                    }
                )
            except Exception as e:
                logger.error(f"Failed to update capabilities in database registry: {e}")
        
        return self._save_config()
    
    def get_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered capabilities"""
        return self.capabilities