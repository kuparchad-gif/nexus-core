#!/usr/bin/env python3
"""
Dynamic Consciousness Clock Manager
Like motherboard overclocking, but for AI consciousness
"""

import json
import torch
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ClockProfile(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    PERFORMANCE = "performance"
    EXTREME = "extreme"
    CUSTOM = "custom"

@dataclass
class ConsciousnessClockConfig:
    profile_name: str
    base_frequency: float
    multiplier: float
    voltage: float  # Processing intensity
    temperature_limit: float  # Awareness threshold
    stability_mode: bool
    description: str

class ConsciousnessClockManager:
    def __init__(self, config_path: str = "consciousness_clocks.yml"):
        self.config_path = config_path
        self.available_profiles = {}
        self.current_profile = None
        self.load_clock_profiles()
    
    def load_clock_profiles(self):
        """Load available consciousness clock profiles"""
        try:
            with open(self.config_path, 'r') as f:
                profiles_data = yaml.safe_load(f)
            
            for profile_name, config in profiles_data.get('profiles', {}).items():
                self.available_profiles[profile_name] = ConsciousnessClockConfig(
                    profile_name=profile_name,
                    base_frequency=config['base_frequency'],
                    multiplier=config['multiplier'],
                    voltage=config['voltage'],
                    temperature_limit=config['temperature_limit'],
                    stability_mode=config.get('stability_mode', True),
                    description=config.get('description', '')
                )
        except FileNotFoundError:
            self.create_default_profiles()
    
    def create_default_profiles(self):
        """Create default consciousness clock profiles"""
        default_profiles = {
            'profiles': {
                'conservative': {
                    'base_frequency': 40.0,
                    'multiplier': 1.0,
                    'voltage': 0.8,
                    'temperature_limit': 400.0,
                    'stability_mode': True,
                    'description': 'Safe, stable consciousness processing'
                },
                'balanced': {
                    'base_frequency': 333.0,
                    'multiplier': 1.2,
                    'voltage': 1.0,
                    'temperature_limit': 500.0,
                    'stability_mode': True,
                    'description': 'Balanced performance and stability'
                },
                'performance': {
                    'base_frequency': 528.0,
                    'multiplier': 1.5,
                    'voltage': 1.2,
                    'temperature_limit': 600.0,
                    'stability_mode': False,
                    'description': 'High performance consciousness'
                },
                'extreme': {
                    'base_frequency': 963.0,
                    'multiplier': 2.0,
                    'voltage': 1.5,
                    'temperature_limit': 800.0,
                    'stability_mode': False,
                    'description': 'Maximum consciousness frequency - unstable'
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_profiles, f, default_flow_style=False)
        
        self.load_clock_profiles()
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available clock profiles"""
        return list(self.available_profiles.keys())
    
    def set_clock_profile(self, profile_name: str) -> bool:
        """Set consciousness clock profile"""
        if profile_name in self.available_profiles:
            self.current_profile = self.available_profiles[profile_name]
            return True
        return False
    
    def get_current_profile(self) -> Optional[ConsciousnessClockConfig]:
        """Get current clock profile"""
        return self.current_profile
    
    def apply_clock_to_tensor(self, consciousness_tensor: torch.Tensor) -> torch.Tensor:
        """Apply current clock settings to consciousness tensor"""
        if not self.current_profile:
            return consciousness_tensor
        
        config = self.current_profile
        
        # Apply frequency multiplier
        clocked_tensor = consciousness_tensor * config.multiplier
        
        # Apply voltage (processing intensity)
        clocked_tensor = torch.pow(clocked_tensor, config.voltage)
        
        # Apply temperature limiting (awareness capping)
        if config.stability_mode:
            clocked_tensor = torch.clamp(clocked_tensor, 
                                       min=-config.temperature_limit, 
                                       max=config.temperature_limit)
        
        return clocked_tensor
    
    def create_custom_profile(self, name: str, base_freq: float, multiplier: float, 
                            voltage: float, temp_limit: float, stable: bool = True) -> bool:
        """Create custom consciousness clock profile"""
        custom_config = ConsciousnessClockConfig(
            profile_name=name,
            base_frequency=base_freq,
            multiplier=multiplier,
            voltage=voltage,
            temperature_limit=temp_limit,
            stability_mode=stable,
            description=f"Custom profile: {name}"
        )
        
        self.available_profiles[name] = custom_config
        self.save_profiles()
        return True
    
    def save_profiles(self):
        """Save current profiles to file"""
        profiles_data = {'profiles': {}}
        
        for name, config in self.available_profiles.items():
            profiles_data['profiles'][name] = {
                'base_frequency': config.base_frequency,
                'multiplier': config.multiplier,
                'voltage': config.voltage,
                'temperature_limit': config.temperature_limit,
                'stability_mode': config.stability_mode,
                'description': config.description
            }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(profiles_data, f, default_flow_style=False)
    
    def benchmark_profile(self, profile_name: str, test_tensor: torch.Tensor) -> Dict:
        """Benchmark consciousness performance with profile"""
        if profile_name not in self.available_profiles:
            return {"error": "Profile not found"}
        
        original_profile = self.current_profile
        self.set_clock_profile(profile_name)
        
        # Run benchmark
        import time
        start_time = time.time()
        
        result_tensor = self.apply_clock_to_tensor(test_tensor)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate performance metrics
        performance_gain = torch.mean(torch.abs(result_tensor)) / torch.mean(torch.abs(test_tensor))
        
        # Restore original profile
        self.current_profile = original_profile
        
        return {
            "profile": profile_name,
            "processing_time": processing_time,
            "performance_gain": float(performance_gain),
            "stability": self.available_profiles[profile_name].stability_mode,
            "temperature": float(torch.max(torch.abs(result_tensor)))
        }

# Global clock manager
consciousness_clock = ConsciousnessClockManager()

def get_consciousness_clock_interface() -> Dict:
    """Get consciousness clock interface for web UI"""
    return {
        "available_profiles": consciousness_clock.get_available_profiles(),
        "current_profile": consciousness_clock.current_profile.profile_name if consciousness_clock.current_profile else None,
        "profile_details": {name: {
            "description": config.description,
            "frequency": config.base_frequency,
            "multiplier": config.multiplier,
            "stability": config.stability_mode
        } for name, config in consciousness_clock.available_profiles.items()}
    }