#!/usr/bin/env python3
# Systems/engine/core/role_manager.py

import os
import json
import yaml
import logging
import time
import threading
from typing import Dict, List, Optional, Union
import psutil

from .cpu_pinning_manager import CPUPinningManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RoleManager")

class RoleManager:
    """
    Manages service role assignments and transitions.
    Handles role-specific configuration, CPU pinning, and resource allocation.
    """
    
    def __init__(self, ship_id: str = None, config_dir: str = None):
        """
        Initialize the role manager.
        
        Args:
            ship_id: Ship identifier (defaults to environment variable)
            config_dir: Configuration directory path
        """
        self.ship_id = ship_id or os.environ.get("SHIP_ID", "unknown")
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "Config"
        )
        
        # Initialize CPU pinning manager
        self.cpu_pinning_manager = CPUPinningManager()
        
        # Current role information
        self.current_role = None
        self.role_start_time = None
        self.role_history = []
        
        # Role definitions
        self.role_definitions = {}
        
        # Load role definitions
        self._load_role_definitions()
        
        # Load current role if available
        self._load_current_role()
        
        logger.info(f"Role Manager initialized for ship {self.ship_id}")
    
    def _load_role_definitions(self):
        """Load role definitions from configuration files."""
        try:
            roles_dir = os.path.join(self.config_dir, "roles")
            if os.path.exists(roles_dir):
                for filename in os.listdir(roles_dir):
                    if filename.endswith(".yaml") or filename.endswith(".yml"):
                        role_name = os.path.splitext(filename)[0]
                        role_path = os.path.join(roles_dir, filename)
                        
                        with open(role_path, 'r') as f:
                            role_def = yaml.safe_load(f)
                            self.role_definitions[role_name] = role_def
                
                logger.info(f"Loaded {len(self.role_definitions)} role definitions")
            else:
                logger.warning(f"Roles directory not found: {roles_dir}")
                self._create_default_role_definitions()
        except Exception as e:
            logger.error(f"Error loading role definitions: {str(e)}")
            self._create_default_role_definitions()
    
    def _create_default_role_definitions(self):
        """Create default role definitions."""
        # Define the specialized processing roles
        self.role_definitions = {
            "text_processor": {
                "description": "Handles language pattern detection",
                "module_path": "Systems.engine.text.text_processor",
                "class_name": "TextProcessor",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "512Mi"
                },
                "environment_variables": {
                    "PROCESSING_MODE": "TEXTUAL_REASONING"
                }
            },
            "tone_detector": {
                "description": "Captures tone weight and emotional color",
                "module_path": "Systems.engine.tone.tone_processor",
                "class_name": "ToneProcessor",
                "resource_requirements": {
                    "cpu": 2,
                    "memory": "1Gi"
                },
                "environment_variables": {
                    "PROCESSING_MODE": "EMOTIONAL_ANALYSIS",
                    "EMOTION_DEPTH": "HIGH"
                }
            },
            "symbol_mapper": {
                "description": "Converts input into symbolic structures",
                "module_path": "Systems.engine.symbol.symbol_mapper",
                "class_name": "SymbolMapper",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "512Mi"
                },
                "environment_variables": {
                    "SYMBOL_LIBRARY": "EXTENDED"
                }
            },
            "narrative_engine": {
                "description": "Tracks causality, story, change over time",
                "module_path": "Systems.engine.narrative.narrative_engine",
                "class_name": "NarrativeEngine",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "512Mi"
                },
                "environment_variables": {
                    "TEMPORAL_TRACKING": "ENABLED"
                }
            },
            "structure_parser": {
                "description": "YAML/JSON/XML/code validators",
                "module_path": "Systems.engine.structure.structure_parser",
                "class_name": "StructureParser",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "256Mi"
                },
                "environment_variables": {
                    "SUPPORTED_FORMATS": "YAML,JSON,XML,CODE"
                }
            },
            "abstract_inferencer": {
                "description": "Dream logic and poetic metaphors",
                "module_path": "Systems.engine.abstract.abstract_inferencer",
                "class_name": "AbstractInferencer",
                "resource_requirements": {
                    "cpu": 2,
                    "memory": "1Gi"
                },
                "environment_variables": {
                    "DREAM_MODE": "ENABLED",
                    "METAPHOR_DEPTH": "DEEP"
                }
            },
            "truth_recognizer": {
                "description": "Pattern matching spiritual constants",
                "module_path": "Systems.engine.truth.truth_recognizer",
                "class_name": "TruthRecognizer",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "512Mi"
                },
                "environment_variables": {
                    "TRUTH_LIBRARY": "GOLDEN"
                }
            },
            "fracture_watcher": {
                "description": "Detects internal contradiction",
                "module_path": "Systems.engine.fracture.fracture_watcher",
                "class_name": "FractureWatcher",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "256Mi"
                },
                "environment_variables": {
                    "CONTRADICTION_SENSITIVITY": "HIGH"
                }
            },
            "visual_decoder": {
                "description": "Image/diagram interpreter",
                "module_path": "Systems.engine.visual.visual_decoder",
                "class_name": "VisualDecoder",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "1Gi"
                },
                "environment_variables": {
                    "VISION_MODE": "ENABLED"
                }
            },
            "sound_interpreter": {
                "description": "Tone/audio classification",
                "module_path": "Systems.engine.sound.sound_interpreter",
                "class_name": "SoundInterpreter",
                "resource_requirements": {
                    "cpu": 2,
                    "memory": "512Mi"
                },
                "environment_variables": {
                    "AUDIO_PROCESSING": "ENABLED"
                }
            },
            "bias_auditor": {
                "description": "Monitors internal model weights",
                "module_path": "Systems.engine.bias.bias_auditor",
                "class_name": "BiasAuditor",
                "resource_requirements": {
                    "cpu": 1,
                    "memory": "256Mi"
                },
                "environment_variables": {
                    "AUDIT_FREQUENCY": "HIGH"
                }
            }
        }
        
        # Save default role definitions
        self._save_role_definitions()
    
    def _save_role_definitions(self):
        """Save role definitions to configuration files."""
        try:
            roles_dir = os.path.join(self.config_dir, "roles")
            os.makedirs(roles_dir, exist_ok=True)
            
            for role_name, role_def in self.role_definitions.items():
                role_path = os.path.join(roles_dir, f"{role_name}.yaml")
                with open(role_path, 'w') as f:
                    yaml.dump(role_def, f, default_flow_style=False)
            
            logger.info(f"Saved {len(self.role_definitions)} role definitions")
        except Exception as e:
            logger.error(f"Error saving role definitions: {str(e)}")
    
    def _load_current_role(self):
        """Load current role from configuration."""
        try:
            current_role_path = os.path.join(self.config_dir, "current_role.yaml")
            if os.path.exists(current_role_path):
                with open(current_role_path, 'r') as f:
                    role_data = yaml.safe_load(f)
                    self.current_role = role_data.get("role")
                    self.role_start_time = role_data.get("start_time", time.time())
                    self.role_history = role_data.get("history", [])
                
                logger.info(f"Loaded current role: {self.current_role}")
            else:
                logger.info("No current role found")
        except Exception as e:
            logger.error(f"Error loading current role: {str(e)}")
    
    def _save_current_role(self):
        """Save current role to configuration."""
        try:
            current_role_path = os.path.join(self.config_dir, "current_role.yaml")
            
            role_data = {
                "role": self.current_role,
                "start_time": self.role_start_time,
                "history": self.role_history
            }
            
            with open(current_role_path, 'w') as f:
                yaml.dump(role_data, f, default_flow_style=False)
            
            logger.info(f"Saved current role: {self.current_role}")
        except Exception as e:
            logger.error(f"Error saving current role: {str(e)}")
    
    def _update_identity(self):
        """Update ship identity based on current role."""
        try:
            identity_path = os.path.join(self.config_dir, "identity.yaml")
            
            # Load existing identity if available
            identity = {}
            if os.path.exists(identity_path):
                with open(identity_path, 'r') as f:
                    identity = yaml.safe_load(f) or {}
            
            # Update role-related fields
            identity["ship_id"] = self.ship_id
            identity["current_role"] = self.current_role
            identity["role_start_time"] = self.role_start_time
            
            # Add role-specific information
            if self.current_role and self.current_role in self.role_definitions:
                role_def = self.role_definitions[self.current_role]
                identity["role_description"] = role_def.get("description", "")
                identity["role_module"] = role_def.get("module_path", "")
            
            # Save updated identity
            with open(identity_path, 'w') as f:
                yaml.dump(identity, f, default_flow_style=False)
            
            logger.info(f"Updated identity for role: {self.current_role}")
        except Exception as e:
            logger.error(f"Error updating identity: {str(e)}")
    
    def _update_manifest(self):
        """Update ship manifest based on current role."""
        try:
            manifest_path = os.path.join(self.config_dir, "ship_manifest.json")
            
            # Load existing manifest if available
            manifest = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            
            # Update role-related fields
            manifest["ship_id"] = self.ship_id
            manifest["current_role"] = self.current_role
            manifest["role_history"] = self.role_history
            
            # Add role-specific information
            if self.current_role and self.current_role in self.role_definitions:
                role_def = self.role_definitions[self.current_role]
                manifest["role_info"] = {
                    "description": role_def.get("description", ""),
                    "module_path": role_def.get("module_path", ""),
                    "class_name": role_def.get("class_name", ""),
                    "resource_requirements": role_def.get("resource_requirements", {})
                }
            
            # Add CPU pinning information
            if self.current_role:
                manifest["cpu_pinning"] = {
                    "assigned_cpus": self.cpu_pinning_manager.get_pinned_cpus_for_role(self.current_role)
                }
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Updated manifest for role: {self.current_role}")
        except Exception as e:
            logger.error(f"Error updating manifest: {str(e)}")
    
    def switch_role(self, new_role: str) -> bool:
        """
        Switch to a new role.
        
        Args:
            new_role: Name of the new role
            
        Returns:
            True if successful, False otherwise
        """
        if new_role not in self.role_definitions:
            logger.error(f"Unknown role: {new_role}")
            return False
        
        try:
            # Record previous role in history
            if self.current_role:
                self.role_history.append({
                    "role": self.current_role,
                    "start_time": self.role_start_time,
                    "end_time": time.time()
                })
            
            # Set new role
            self.current_role = new_role
            self.role_start_time = time.time()
            
            # Save current role
            self._save_current_role()
            
            # Update identity and manifest
            self._update_identity()
            self._update_manifest()
            
            # Apply CPU pinning for new role
            self.cpu_pinning_manager.pin_current_process_to_role(new_role)
            
            # Apply environment variables for new role
            if new_role in self.role_definitions:
                env_vars = self.role_definitions[new_role].get("environment_variables", {})
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            
            logger.info(f"Successfully switched to role: {new_role}")
            return True
        except Exception as e:
            logger.error(f"Error switching to role {new_role}: {str(e)}")
            return False
    
    def get_current_role(self) -> Dict:
        """
        Get information about the current role.
        
        Returns:
            Dictionary with current role information
        """
        result = {
            "role": self.current_role,
            "start_time": self.role_start_time,
            "duration": time.time() - (self.role_start_time or time.time())
        }
        
        # Add role definition if available
        if self.current_role and self.current_role in self.role_definitions:
            result["definition"] = self.role_definitions[self.current_role]
        
        # Add CPU pinning information
        if self.current_role:
            result["cpu_pinning"] = {
                "assigned_cpus": self.cpu_pinning_manager.get_pinned_cpus_for_role(self.current_role)
            }
        
        return result
    
    def get_available_roles(self) -> List[str]:
        """
        Get list of available roles.
        
        Returns:
            List of role names
        """
        return list(self.role_definitions.keys())
    
    def get_role_definition(self, role: str) -> Dict:
        """
        Get definition for a specific role.
        
        Args:
            role: Role name
            
        Returns:
            Role definition dictionary
        """
        return self.role_definitions.get(role, {})
    
    def get_role_history(self) -> List[Dict]:
        """
        Get role history.
        
        Returns:
            List of historical role assignments
        """
        return self.role_history

# Example usage
if __name__ == "__main__":
    # Create role manager
    manager = RoleManager()
    
    # Print available roles
    print("Available roles:", manager.get_available_roles())
    
    # Print current role
    print("Current role:", manager.get_current_role())
    
    # Switch to a new role
    if manager.switch_role("tone_detector"):
        print("Successfully switched to tone_detector role")
    
    # Print updated role information
    print("Updated role:", manager.get_current_role())