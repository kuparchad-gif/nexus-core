#!/usr/bin/env python3
"""
Dynamic Network Adapter for Viren Services
Handles port conflicts, service discovery, and dynamic configuration
"""

import yaml
import os
import socket
import requests
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("NetworkAdapter")

class VirenNetworkAdapter:
    def __init__(self, config_path: str = "viren_network_config.yml", env_path: str = "Config\\.well-known\\.env"):
        self.config_path = config_path
        self.env_path = env_path
        self.config = self.load_config()
        self.service_registry = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load dynamic network configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()
    
    def _substitute_env_vars(self, obj):
        """Recursively substitute environment variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            # Extract env var with default: ${VAR_NAME:-default}
            env_expr = obj[2:-1]  # Remove ${ and }
            if ':-' in env_expr:
                var_name, default = env_expr.split(':-', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(env_expr, obj)
        return obj
    
    def _default_config(self) -> Dict[str, Any]:
        """Fallback configuration"""
        return {
            "services": {
                "master_horn": {"internal_port": 333, "external_port": 333},
                "gabriel_horn": {"internal_port": 7860, "external_port": 7860},
                "viren_api": {"internal_port": 8081, "external_port": 8081},
                "viren_bridge": {"internal_port": 8082, "external_port": 8082},
                "viren_portal": {"internal_port": 8083, "external_port": 8083}
            }
        }
    
    def check_port_available(self, port: int, host: str = "localhost") -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result != 0  # Port is available if connection fails
        except:
            return False
    
    def find_available_port(self, preferred_port: int, start_range: int = None) -> int:
        """Find next available port"""
        if self.check_port_available(preferred_port):
            return preferred_port
        
        # Search in range
        start = start_range or (preferred_port + 1)
        for port in range(start, start + 100):
            if self.check_port_available(port):
                logger.info(f"Port {preferred_port} unavailable, using {port}")
                return port
        
        raise Exception(f"No available ports found near {preferred_port}")
    
    def resolve_service_ports(self) -> Dict[str, Dict[str, int]]:
        """Resolve all service ports, handling conflicts"""
        resolved_ports = {}
        
        for service_name, service_config in self.config["services"].items():
            external_port = service_config["external_port"]
            available_port = self.find_available_port(external_port)
            
            resolved_ports[service_name] = {
                "internal_port": service_config["internal_port"],
                "external_port": available_port,
                "host": service_config.get("host", service_name)
            }
            
            if available_port != external_port:
                logger.warning(f"{service_name}: Port {external_port} → {available_port}")
        
        return resolved_ports
    
    def generate_docker_compose_env(self) -> str:
        """Generate environment variables for docker-compose"""
        resolved_ports = self.resolve_service_ports()
        env_vars = []
        
        for service_name, ports in resolved_ports.items():
            service_upper = service_name.upper().replace('-', '_')
            env_vars.append(f"{service_upper}_EXTERNAL_PORT={ports['external_port']}")
            env_vars.append(f"{service_upper}_INTERNAL_PORT={ports['internal_port']}")
            env_vars.append(f"{service_upper}_HOST={ports['host']}")
        
        return '\n'.join(env_vars)
    
    def get_service_url(self, service_name: str, external: bool = False) -> str:
        """Get service URL for communication"""
        service_config = self.config["services"].get(service_name)
        if not service_config:
            raise ValueError(f"Service {service_name} not found in config")
        
        host = service_config["host"]
        port = service_config["external_port"] if external else service_config["internal_port"]
        
        return f"http://{host}:{port}"
    
    def health_check_service(self, service_name: str) -> bool:
        """Check if service is healthy"""
        try:
            url = self.get_service_url(service_name)
            health_endpoint = self.config["services"][service_name].get("health_endpoint", "/health")
            
            response = requests.get(f"{url}{health_endpoint}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_service(self, service_name: str, timeout: int = 60) -> bool:
        """Wait for service to become healthy"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.health_check_service(service_name):
                logger.info(f"✅ {service_name} is healthy")
                return True
            time.sleep(2)
        
        logger.error(f"❌ {service_name} failed to become healthy")
        return False

# Global adapter instance
network_adapter = VirenNetworkAdapter()

def get_service_url(service_name: str, external: bool = False) -> str:
    """Convenience function for getting service URLs"""
    return network_adapter.get_service_url(service_name, external)

def check_all_services_healthy() -> Dict[str, bool]:
    """Check health of all services"""
    health_status = {}
    for service_name in network_adapter.config["services"].keys():
        health_status[service_name] = network_adapter.health_check_service(service_name)
    return health_status