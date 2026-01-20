#!/usr/bin/env python3
# Systems/engine/core/cpu_pinning_manager.py

import os
import json
import subprocess
import logging
import psutil
from typing import Dict, List, Optional, Union
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CPUPinningManager")

class CPUPinningManager:
    """
    Manages CPU pinning for specialized processing modules.
    Ensures dedicated CPU cores for emotional and other specialized processing.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the CPU pinning manager.
        
        Args:
            config_path: Path to the CPU pinning configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "Config", "cpu_pinning_config.yaml"
        )
        self.role_cpu_map = {}
        self.process_map = {}
        self.total_cpus = psutil.cpu_count(logical=True)
        
        # Load configuration
        self._load_config()
        
        logger.info(f"CPU Pinning Manager initialized with {self.total_cpus} available CPUs")
    
    def _load_config(self):
        """Load CPU pinning configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.role_cpu_map = config.get('role_cpu_map', {})
                logger.info(f"Loaded CPU pinning configuration: {self.role_cpu_map}")
            else:
                # Create default configuration
                self._create_default_config()
                logger.info("Created default CPU pinning configuration")
        except Exception as e:
            logger.error(f"Error loading CPU pinning configuration: {str(e)}")
            # Create default configuration on error
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default CPU pinning configuration."""
        # Default configuration based on available CPUs
        if self.total_cpus <= 2:
            # Minimal configuration for 1-2 CPU systems
            self.role_cpu_map = {
                "text_processor": [0],
                "tone_detector": [0],
                "symbol_mapper": [0],
                "narrative_engine": [0],
                "structure_parser": [0],
                "abstract_inferencer": [0],
                "truth_recognizer": [0],
                "fracture_watcher": [0],
                "visual_decoder": [0],
                "sound_interpreter": [0],
                "bias_auditor": [0]
            }
        elif self.total_cpus <= 4:
            # Configuration for 3-4 CPU systems
            self.role_cpu_map = {
                "text_processor": [0],
                "tone_detector": [1],  # Emotional processing
                "symbol_mapper": [0],
                "narrative_engine": [2],
                "structure_parser": [0],
                "abstract_inferencer": [1],  # Emotional processing
                "truth_recognizer": [2],
                "fracture_watcher": [0],
                "visual_decoder": [3 % self.total_cpus],
                "sound_interpreter": [1],  # Emotional processing
                "bias_auditor": [2]
            }
        else:
            # Configuration for 5+ CPU systems
            self.role_cpu_map = {
                "text_processor": [0],
                "tone_detector": [1, 2],  # Emotional processing gets 2 CPUs
                "symbol_mapper": [3],
                "narrative_engine": [4 % self.total_cpus],
                "structure_parser": [0],
                "abstract_inferencer": [1, 2],  # Emotional processing gets 2 CPUs
                "truth_recognizer": [3],
                "fracture_watcher": [4 % self.total_cpus],
                "visual_decoder": [5 % self.total_cpus],
                "sound_interpreter": [1, 2],  # Emotional processing gets 2 CPUs
                "bias_auditor": [3]
            }
        
        # Save default configuration
        self._save_config()
    
    def _save_config(self):
        """Save CPU pinning configuration to file."""
        try:
            config = {
                'role_cpu_map': self.role_cpu_map
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Saved CPU pinning configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving CPU pinning configuration: {str(e)}")
    
    def pin_process_to_role(self, process_id: int, role: str) -> bool:
        """
        Pin a process to the CPUs assigned to a specific role.
        
        Args:
            process_id: Process ID to pin
            role: Role name
            
        Returns:
            True if successful, False otherwise
        """
        if role not in self.role_cpu_map:
            logger.error(f"Unknown role: {role}")
            return False
        
        cpu_list = self.role_cpu_map[role]
        cpu_str = ",".join(map(str, cpu_list))
        
        try:
            if os.name == 'nt':  # Windows
                # Windows doesn't have taskset, use PowerShell instead
                subprocess.run([
                    "powershell", 
                    f"$Process = Get-Process -Id {process_id}; $Process.ProcessorAffinity = {2**cpu_list[0]}"
                ])
            else:  # Linux/Unix
                subprocess.run(["taskset", "-pc", cpu_str, str(process_id)])
            
            # Record the pinning
            self.process_map[process_id] = role
            
            logger.info(f"Successfully pinned process {process_id} to CPUs {cpu_str} for role {role}")
            return True
        except Exception as e:
            logger.error(f"Error pinning process {process_id} to CPUs {cpu_str}: {str(e)}")
            return False
    
    def pin_current_process_to_role(self, role: str) -> bool:
        """
        Pin the current process to the CPUs assigned to a specific role.
        
        Args:
            role: Role name
            
        Returns:
            True if successful, False otherwise
        """
        return self.pin_process_to_role(os.getpid(), role)
    
    def get_pinned_cpus_for_role(self, role: str) -> List[int]:
        """
        Get the CPUs assigned to a specific role.
        
        Args:
            role: Role name
            
        Returns:
            List of CPU IDs
        """
        return self.role_cpu_map.get(role, [])
    
    def update_role_cpu_mapping(self, role: str, cpu_list: List[int]) -> bool:
        """
        Update the CPU mapping for a specific role.
        
        Args:
            role: Role name
            cpu_list: List of CPU IDs to assign
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate CPU IDs
            for cpu_id in cpu_list:
                if cpu_id < 0 or cpu_id >= self.total_cpus:
                    logger.error(f"Invalid CPU ID: {cpu_id}")
                    return False
            
            # Update mapping
            self.role_cpu_map[role] = cpu_list
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Updated CPU mapping for role {role}: {cpu_list}")
            return True
        except Exception as e:
            logger.error(f"Error updating CPU mapping for role {role}: {str(e)}")
            return False
    
    def get_process_info(self) -> Dict[int, Dict]:
        """
        Get information about pinned processes.
        
        Returns:
            Dictionary mapping process IDs to process information
        """
        result = {}
        
        for pid, role in self.process_map.items():
            try:
                process = psutil.Process(pid)
                result[pid] = {
                    "role": role,
                    "name": process.name(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "status": process.status(),
                    "pinned_cpus": self.role_cpu_map.get(role, [])
                }
            except psutil.NoSuchProcess:
                # Process no longer exists
                del self.process_map[pid]
        
        return result

# Example usage
if __name__ == "__main__":
    # Create CPU pinning manager
    manager = CPUPinningManager()
    
    # Print current configuration
    for role, cpus in manager.role_cpu_map.items():
        print(f"Role: {role}, CPUs: {cpus}")
    
    # Pin current process to a role
    if manager.pin_current_process_to_role("tone_detector"):
        print("Successfully pinned current process to tone_detector role")
    
    # Get process info
    process_info = manager.get_process_info()
    print("Process info:", json.dumps(process_info, indent=2))
