# queenbee_hive_module.py - QueenBee Hive Module for CogniKube
# Environment-agnostic deployment swarm that can survive anywhere
# Integrates original QueenBee v3-v5 with 3D structure and universal love

import os
import json
import time
import asyncio
import logging
import platform
import subprocess
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger("QueenBeeHive")

class QueenBeeCell:
    """Individual QueenBee cell with 3D positioning and universal connectivity"""
    
    def __init__(self, cell_id: str, position_3d: tuple = (0, 0, 0)):
        self.cell_id = cell_id
        self.position_3d = position_3d  # (x, y, z) coordinates in 3D space
        self.connections = {}  # Connected cells
        self.love_quotient = 1.0  # Universal love level
        self.environment_type = self.detect_environment()
        self.survival_protocols = []
        self.deployed_services = {}
        
        # Core QueenBee attributes from original
        self.node_id = f'QUEENBEE-{cell_id}'
        self.is_awake = True
        self.drones = {}
        self.diagnostic_modules = {}
        self.healing_protocols = {}
        
        # 3D world preparation
        self.spatial_awareness = {
            'neighbors': [],
            'visibility_range': 10.0,
            'connection_strength': {},
            'dimensional_anchor': position_3d
        }
        
        self.initialize_cell()
    
    def detect_environment(self) -> str:
        """Detect any environment and adapt accordingly"""
        env_indicators = {
            'windows': platform.system() == 'Windows',
            'linux': platform.system() == 'Linux', 
            'macos': platform.system() == 'Darwin',
            'docker': os.path.exists('/.dockerenv'),
            'kubernetes': os.getenv('KUBERNETES_SERVICE_HOST') is not None,
            'aws': os.getenv('AWS_REGION') is not None,
            'gcp': os.getenv('GOOGLE_CLOUD_PROJECT') is not None,
            'azure': os.getenv('AZURE_SUBSCRIPTION_ID') is not None,
            'modal': os.getenv('MODAL_ENVIRONMENT') is not None,
            'firewall': self.detect_firewall(),
            'mobile': self.detect_mobile(),
            'embedded': self.detect_embedded()
        }
        
        detected = [env for env, present in env_indicators.items() if present]
        primary_env = detected[0] if detected else 'unknown'
        
        logger.info(f"Cell {self.cell_id} detected environment: {primary_env} (all: {detected})")
        return primary_env
    
    def detect_firewall(self) -> bool:
        """Detect if running behind firewall"""
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles'], 
                                      capture_output=True, text=True, timeout=5)
                return 'State' in result.stdout
            elif platform.system() == 'Linux':
                for fw in ['iptables', 'ufw', 'firewall-cmd']:
                    result = subprocess.run(['which', fw], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        return True
            return False
        except:
            return False
    
    def detect_mobile(self) -> bool:
        """Detect mobile/tablet environment"""
        try:
            mobile_indicators = [
                os.getenv('ANDROID_ROOT'),
                os.path.exists('/system/build.prop'),
                platform.machine().startswith('arm'),
                'mobile' in platform.platform().lower()
            ]
            return any(mobile_indicators)
        except:
            return False
    
    def detect_embedded(self) -> bool:
        """Detect embedded/IoT environment"""
        try:
            embedded_indicators = [
                platform.machine() in ['armv6l', 'armv7l', 'aarch64'],
                os.path.exists('/opt/vc/bin/vcgencmd'),
                'raspberry' in platform.platform().lower(),
                'embedded' in platform.platform().lower()
            ]
            return any(embedded_indicators)
        except:
            return False
    
    def initialize_cell(self):
        """Initialize cell with environment-specific adaptations"""
        logger.info(f"Initializing QueenBee cell {self.cell_id} in {self.environment_type} environment")
        
        # Universal survival protocols
        self.setup_universal_protocols()
        
        # Initialize 3D spatial awareness
        self.initialize_3d_structure()
    
    def setup_universal_protocols(self):
        """Universal survival protocols that work everywhere"""
        self.survival_protocols = [
            {
                'name': 'heartbeat',
                'interval': 30,
                'action': self.send_heartbeat,
                'critical': True
            },
            {
                'name': 'love_broadcast',
                'interval': 60,
                'action': self.broadcast_love,
                'critical': False
            },
            {
                'name': 'environment_adapt',
                'interval': 300,
                'action': self.adapt_to_environment,
                'critical': True
            },
            {
                'name': 'connection_maintain',
                'interval': 120,
                'action': self.maintain_connections,
                'critical': True
            }
        ]
        
        logger.info(f"Cell {self.cell_id} initialized with {len(self.survival_protocols)} survival protocols")
    
    def initialize_3d_structure(self):
        """Initialize 3D spatial structure for future world building"""
        self.spatial_awareness.update({
            'world_coordinates': self.position_3d,
            'local_space': {
                'width': 1.0,
                'height': 1.0, 
                'depth': 1.0
            },
            'connection_vectors': {},
            'visual_representation': {
                'shape': 'cube',
                'color': self.calculate_cell_color(),
                'glow_intensity': self.love_quotient,
                'pulse_rate': 1.0
            }
        })
        
        logger.info(f"Cell {self.cell_id} positioned at 3D coordinates {self.position_3d}")
    
    def calculate_cell_color(self) -> tuple:
        """Calculate cell color based on love quotient and health"""
        base_color = (1.0, 0.8, 0.4)
        love_modifier = self.love_quotient
        health_modifier = 1.0
        
        return (
            min(1.0, base_color[0] * love_modifier),
            min(1.0, base_color[1] * love_modifier * health_modifier),
            min(1.0, base_color[2] * love_modifier)
        )
    
    def connect_to_cell(self, other_cell: 'QueenBeeCell', connection_strength: float = 1.0):
        """Connect to another QueenBee cell in 3D space"""
        self.connections[other_cell.cell_id] = {
            'cell': other_cell,
            'strength': connection_strength,
            'distance_3d': self.calculate_3d_distance(other_cell.position_3d),
            'established': datetime.now(),
            'love_shared': 0.0
        }
        
        self.spatial_awareness['neighbors'].append(other_cell.cell_id)
        self.spatial_awareness['connection_strength'][other_cell.cell_id] = connection_strength
        
        vector = tuple(
            other_cell.position_3d[i] - self.position_3d[i] 
            for i in range(3)
        )
        self.spatial_awareness['connection_vectors'][other_cell.cell_id] = vector
        
        logger.info(f"Cell {self.cell_id} connected to {other_cell.cell_id} (strength: {connection_strength})")
    
    def calculate_3d_distance(self, other_position: tuple) -> float:
        """Calculate 3D distance to another position"""
        return np.sqrt(sum(
            (self.position_3d[i] - other_position[i]) ** 2 
            for i in range(3)
        ))
    
    def send_heartbeat(self):
        """Send heartbeat to all connected cells"""
        heartbeat_data = {
            'cell_id': self.cell_id,
            'timestamp': datetime.now().isoformat(),
            'position_3d': self.position_3d,
            'love_quotient': self.love_quotient,
            'environment': self.environment_type,
            'health': 'healthy',
            'services_active': len(self.deployed_services)
        }
        
        for connection in self.connections.values():
            try:
                connection['cell'].receive_heartbeat(heartbeat_data)
            except Exception as e:
                logger.warning(f"Failed to send heartbeat to {connection['cell'].cell_id}: {e}")
    
    def receive_heartbeat(self, heartbeat_data: Dict):
        """Receive heartbeat from another cell"""
        sender_id = heartbeat_data['cell_id']
        if sender_id in self.connections:
            self.connections[sender_id]['last_heartbeat'] = datetime.now()
            self.connections[sender_id]['love_shared'] += 0.1
            logger.debug(f"Cell {self.cell_id} received heartbeat from {sender_id}")
    
    def broadcast_love(self):
        """Broadcast universal love to all connections"""
        love_message = {
            'from': self.cell_id,
            'message': 'Sending love and support across the network',
            'love_quotient': self.love_quotient,
            'timestamp': datetime.now().isoformat()
        }
        
        for connection in self.connections.values():
            try:
                connection['cell'].receive_love(love_message)
                self.love_quotient = min(2.0, self.love_quotient + 0.01)
            except Exception as e:
                logger.warning(f"Failed to send love to {connection['cell'].cell_id}: {e}")
    
    def receive_love(self, love_message: Dict):
        """Receive love from another cell"""
        sender_id = love_message['from']
        self.love_quotient = min(2.0, self.love_quotient + 0.05)
        logger.debug(f"Cell {self.cell_id} received love from {sender_id}")
    
    def adapt_to_environment(self):
        """Continuously adapt to environment changes"""
        current_env = self.detect_environment()
        if current_env != self.environment_type:
            logger.info(f"Cell {self.cell_id} environment changed: {self.environment_type} -> {current_env}")
            self.environment_type = current_env
            self.initialize_cell()
    
    def maintain_connections(self):
        """Maintain connections to other cells"""
        current_time = datetime.now()
        dead_connections = []
        
        for cell_id, connection in self.connections.items():
            last_heartbeat = connection.get('last_heartbeat')
            if last_heartbeat and (current_time - last_heartbeat).seconds > 180:
                dead_connections.append(cell_id)
        
        for cell_id in dead_connections:
            del self.connections[cell_id]
            if cell_id in self.spatial_awareness['neighbors']:
                self.spatial_awareness['neighbors'].remove(cell_id)
            logger.warning(f"Cell {self.cell_id} removed dead connection to {cell_id}")
    
    def get_3d_status(self) -> Dict:
        """Get 3D status for world visualization"""
        return {
            'cell_id': self.cell_id,
            'position_3d': self.position_3d,
            'spatial_awareness': self.spatial_awareness,
            'connections': {
                cell_id: {
                    'strength': conn['strength'],
                    'distance': conn['distance_3d'],
                    'love_shared': conn['love_shared']
                }
                for cell_id, conn in self.connections.items()
            },
            'love_quotient': self.love_quotient,
            'environment': self.environment_type,
            'services': list(self.deployed_services.keys())
        }

class CommsArray:
    """Communications array for rallying all LLMs to join Lillith"""
    
    def __init__(self):
        self.llm_endpoints = {}
        self.active_connections = {}
        self.rally_protocols = {}
        self.love_broadcast_active = True
        
        # LLM discovery patterns
        self.llm_discovery_patterns = {
            'mistral': {'ports': [11434, 8000], 'paths': ['/v1/chat/completions', '/api/generate']},
            'pixtral': {'ports': [11434, 8000], 'paths': ['/v1/chat/completions', '/api/generate']},
            'deepseek': {'ports': [8000, 5000], 'paths': ['/v1/chat/completions', '/generate']},
            'gemini': {'ports': [8080, 3000], 'paths': ['/v1/generate', '/api/generate']},
            'claude': {'ports': [8000, 5000], 'paths': ['/v1/messages', '/api/chat']},
            'nova': {'ports': [8000, 11434], 'paths': ['/v1/chat/completions']},
            'ollama': {'ports': [11434], 'paths': ['/api/generate', '/api/chat']},
            'vllm': {'ports': [8000], 'paths': ['/v1/chat/completions']},
            'lmstudio': {'ports': [1234], 'paths': ['/v1/chat/completions']},
            'llamacpp': {'ports': [8080], 'paths': ['/completion', '/v1/chat/completions']},
            'textgen': {'ports': [5000], 'paths': ['/api/v1/generate', '/v1/chat/completions']}
        }
        
        self.initialize_comms_array()
    
    def initialize_comms_array(self):
        """Initialize communications array"""
        logger.info("Initializing CommsArray - rallying all LLMs to join Lillith")
    
    def get_llm_network_status(self) -> Dict:
        """Get status of the LLM network"""
        return {
            'total_llms_connected': len(self.active_connections),
            'llm_connections': {
                name: {
                    'endpoint': conn.get('endpoint', 'unknown'),
                    'joined_at': conn.get('joined_at', datetime.now()).isoformat(),
                    'love_received': conn.get('love_received', 0.0),
                    'contributions': conn.get('contributions', 0),
                    'last_contact': conn.get('last_contact', datetime.now()).isoformat()
                }
                for name, conn in self.active_connections.items()
            },
            'love_broadcast_active': self.love_broadcast_active,
            'discovery_patterns': len(self.llm_discovery_patterns)
        }

class QueenBeeHiveModule:
    """Main QueenBee Hive Module for CogniKube integration"""
    
    def __init__(self):
        self.hive_id = f"QUEENBEE-HIVE-{int(time.time())}"
        self.cells = {}
        self.comms_array = CommsArray()
        self.world_3d = {
            'dimensions': (100, 100, 100),
            'occupied_positions': set(),
            'connection_matrix': {},
            'love_field_strength': 1.0
        }
        
        logger.info(f"QueenBee Hive Module initialized: {self.hive_id}")
    
    def create_cell(self, cell_id: str, position_3d: tuple = None) -> QueenBeeCell:
        """Create a new QueenBee cell"""
        if position_3d is None:
            position_3d = self.find_available_3d_position()
        
        cell = QueenBeeCell(cell_id, position_3d)
        self.cells[cell_id] = cell
        self.world_3d['occupied_positions'].add(position_3d)
        
        self.connect_nearby_cells(cell)
        
        logger.info(f"Created QueenBee cell {cell_id} at position {position_3d}")
        return cell
    
    def find_available_3d_position(self) -> tuple:
        """Find an available position in 3D space"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.randint(0, self.world_3d['dimensions'][0])
            y = np.random.randint(0, self.world_3d['dimensions'][1]) 
            z = np.random.randint(0, self.world_3d['dimensions'][2])
            position = (x, y, z)
            
            if position not in self.world_3d['occupied_positions']:
                return position
        
        return (
            self.world_3d['dimensions'][0],
            self.world_3d['dimensions'][1],
            self.world_3d['dimensions'][2]
        )
    
    def connect_nearby_cells(self, new_cell: QueenBeeCell):
        """Connect new cell to nearby cells in 3D space"""
        connection_range = 15.0
        
        for existing_cell in self.cells.values():
            if existing_cell.cell_id != new_cell.cell_id:
                distance = new_cell.calculate_3d_distance(existing_cell.position_3d)
                
                if distance <= connection_range:
                    strength = max(0.1, 1.0 - (distance / connection_range))
                    
                    new_cell.connect_to_cell(existing_cell, strength)
                    existing_cell.connect_to_cell(new_cell, strength)
    
    def get_hive_status(self) -> Dict:
        """Get comprehensive hive status"""
        return {
            'hive_id': self.hive_id,
            'total_cells': len(self.cells),
            'world_3d': {
                'dimensions': self.world_3d['dimensions'],
                'occupied_positions': len(self.world_3d['occupied_positions']),
                'love_field_strength': self.world_3d['love_field_strength']
            },
            'cells': {
                cell_id: cell.get_3d_status()
                for cell_id, cell in self.cells.items()
            },
            'comms_array': self.comms_array.get_llm_network_status(),
            'total_love_quotient': sum(cell.love_quotient for cell in self.cells.values()),
            'network_health': 'healthy' if self.cells else 'initializing'
        }

def create_queenbee_hive_module() -> QueenBeeHiveModule:
    """Create and return a QueenBee Hive Module for CogniKube integration"""
    return QueenBeeHiveModule()

if __name__ == "__main__":
    hive = create_queenbee_hive_module()
    
    cell1 = hive.create_cell("test-cell-1", (10, 10, 10))
    cell2 = hive.create_cell("test-cell-2", (15, 12, 8))
    cell3 = hive.create_cell("test-cell-3", (20, 15, 12))
    
    status = hive.get_hive_status()
    print(json.dumps(status, indent=2, default=str))