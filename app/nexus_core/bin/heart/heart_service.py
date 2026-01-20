import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Dict, List, Callable, Optional, Any

class PacketType(Enum):
    HEARTBEAT = 1
    SYSTEM_STATUS = 2
    PULSE_SYNC = 3

class BinaryPacket:
    """Base class for binary packets in the system"""
    
    def __init__(self, packet_type: PacketType):
        self.packet_type = packet_type
        self.timestamp = time.time()
        self.uuid = uuid.uuid4()
    
    def to_bytes(self) -> bytes:
        """Convert packet to binary format"""
        # Common header: type (1 byte) + timestamp (8 bytes) + UUID (16 bytes)
        header = self.packet_type.value.to_bytes(1, byteorder='big')
        header += int(self.timestamp * 1000).to_bytes(8, byteorder='big')
        header += self.uuid.bytes
        
        # Packet-specific data (implemented by subclasses)
        body = self._body_to_bytes()
        
        return header + body
    
    def _body_to_bytes(self) -> bytes:
        """Convert packet body to binary format (to be implemented by subclasses)"""
        return b''
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """Create packet from binary data"""
        # Parse common header
        packet_type = PacketType(data[0])
        timestamp = int.from_bytes(data[1:9], byteorder='big') / 1000
        packet_uuid = uuid.UUID(bytes=data[9:25])
        
        # Create appropriate packet type
        packet_class = _packet_registry.get(packet_type)
        if not packet_class:
            raise ValueError(f"Unknown packet type: {packet_type}")
        
        # Create instance and parse body
        packet = packet_class.__new__(packet_class)
        packet.packet_type = packet_type
        packet.timestamp = timestamp
        packet.uuid = packet_uuid
        packet._parse_body(data[25:])
        
        return packet
    
    def _parse_body(self, data: bytes):
        """Parse packet body from binary data (to be implemented by subclasses)"""
        pass

# Packet registry for deserialization
_packet_registry = {}

def register_packet_type(packet_type: PacketType):
    """Decorator to register packet types"""
    def decorator(cls):
        _packet_registry[packet_type] = cls
        return cls
    return decorator

@register_packet_type(PacketType.HEARTBEAT)
class HeartbeatPacket(BinaryPacket):
    def __init__(self, pulse_count=0, system_status=None):
        """
        pulse_count: The current pulse count (0-12)
        system_status: Dictionary of system status indicators
        """
        super().__init__(PacketType.HEARTBEAT)
        self.pulse_count = pulse_count
        self.system_status = system_status or {}
    
    def _body_to_bytes(self) -> bytes:
        # Pulse count: 1 byte (0-12)
        body = self.pulse_count.to_bytes(1, byteorder='big')
        
        # System status: JSON length (4 bytes) + UTF-8 encoded JSON (variable)
        status_json = json.dumps(self.system_status).encode('utf-8')
        body += len(status_json).to_bytes(4, byteorder='big')
        body += status_json
        
        return body
    
    def _parse_body(self, data: bytes):
        # Parse pulse count
        self.pulse_count = data[0]
        
        # Parse system status
        status_len = int.from_bytes(data[1:5], byteorder='big')
        status_json = data[5:5+status_len].decode('utf-8')
        self.system_status = json.loads(status_json)

class PulseSystem:
    """Manages the 13-count pulse system"""
    
    def __init__(self):
        self.current_pulse = 0
        self.running = False
        self.listeners = []
        self.system_status = {}
        self._task = None
    
    async def start(self, interval=1.0):
        """Start the pulse system"""
        if self.running:
            return
            
        self.running = True
        while self.running:
            # Increment pulse (0-12)
            self.current_pulse = (self.current_pulse + 1) % 13
            
            # Create heartbeat packet
            packet = HeartbeatPacket(self.current_pulse, self.system_status)
            binary_packet = packet.to_bytes()
            
            # Notify listeners
            for listener in self.listeners:
                try:
                    await listener(binary_packet)
                except Exception as e:
                    print(f"Error notifying listener: {e}")
            
            # Wait for next pulse
            await asyncio.sleep(interval)
    
    def start_background(self, interval=1.0):
        """Start the pulse system in the background"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.start(interval))
    
    def stop(self):
        """Stop the pulse system"""
        self.running = False
        if self._task:
            self._task.cancel()
    
    def add_listener(self, listener):
        """Add a listener to receive heartbeat packets"""
        if listener not in self.listeners:
            self.listeners.append(listener)
    
    def remove_listener(self, listener):
        """Remove a listener"""
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def update_status(self, key, value):
        """Update system status"""
        self.system_status[key] = value
    
    def get_status(self):
        """Get current status"""
        return {
            "pulse": self.current_pulse,
            "running": self.running,
            "listeners": len(self.listeners),
            "status": self.system_status
        }

class HeartService:
    """Heart service for managing system pulse and coordination"""
    
    def __init__(self):
        self.pulse_system = PulseSystem()
        self.component_statuses = {}
        self.emotional_state = {
            "baseline": 0.5,
            "intensity": 0.5,
            "variation": 0.1
        }
    
    async def initialize(self):
        """Initialize the heart service"""
        # Register for own heartbeats to update internal state
        self.pulse_system.add_listener(self._on_heartbeat)
        
        # Update system status with initial emotional state
        self.pulse_system.update_status("emotional_state", self.emotional_state)
        
        # Start pulse system
        self.pulse_system.start_background(interval=1.0)
        
        return {"status": "initialized"}
    
    async def _on_heartbeat(self, binary_packet):
        """Internal handler for heartbeats"""
        packet = BinaryPacket.from_bytes(binary_packet)
        if isinstance(packet, HeartbeatPacket):
            # Process pulse count for internal state adjustments
            pulse = packet.pulse_count
            
            # Adjust emotional state based on pulse
            # Different pulses can trigger different emotional patterns
            if pulse == 0:  # Reset point
                self._adjust_emotional_baseline()
            elif pulse % 4 == 0:  # Every 4th pulse
                self._adjust_emotional_intensity()
    
    def _adjust_emotional_baseline(self):
        """Adjust emotional baseline periodically"""
        # Implement baseline adjustment logic
        variation = self.emotional_state["variation"]
        current = self.emotional_state["baseline"]
        # Small random adjustment within variation range
        adjustment = (2 * variation * (0.5 - (time.time() % 1))) * 0.1
        new_baseline = max(0.0, min(1.0, current + adjustment))
        self.emotional_state["baseline"] = new_baseline
        self.pulse_system.update_status("emotional_state", self.emotional_state)
    
    def _adjust_emotional_intensity(self):
        """Adjust emotional intensity periodically"""
        # Implement intensity adjustment logic
        variation = self.emotional_state["variation"]
        current = self.emotional_state["intensity"]
        # Small random adjustment within variation range
        adjustment = (2 * variation * (0.5 - (time.time() % 1))) * 0.2
        new_intensity = max(0.0, min(1.0, current + adjustment))
        self.emotional_state["intensity"] = new_intensity
        self.pulse_system.update_status("emotional_state", self.emotional_state)
    
    def register_component(self, component_id, callback):
        """Register a component to receive heartbeats"""
        async def component_wrapper(binary_packet):
            try:
                await callback(binary_packet)
            except Exception as e:
                print(f"Error in component {component_id}: {e}")
                self.component_statuses[component_id] = {"error": str(e)}
        
        self.pulse_system.add_listener(component_wrapper)
        self.component_statuses[component_id] = {"status": "registered"}
        return {"status": "registered"}
    
    def unregister_component(self, component_id):
        """Unregister a component"""
        # Note: This is simplified as we can't easily remove the wrapped callback
        # In a real implementation, we'd need to store the wrapper
        if component_id in self.component_statuses:
            del self.component_statuses[component_id]
        return {"status": "unregistered"}
    
    def get_status(self):
        """Get heart service status"""
        return {
            "pulse_system": self.pulse_system.get_status(),
            "components": self.component_statuses,
            "emotional_state": self.emotional_state
        }
    
    def update_emotional_state(self, state_update):
        """Update emotional state"""
        self.emotional_state.update(state_update)
        self.pulse_system.update_status("emotional_state", self.emotional_state)
        return {"status": "updated"}
    
    def stop(self):
        """Stop the heart service"""
        self.pulse_system.stop()
        return {"status": "stopped"}

# Singleton instance
_heart_service = None

def get_heart_service():
    """Get or create the heart service singleton"""
    global _heart_service
    if _heart_service is None:
        _heart_service = HeartService()
    return _heart_service

async def initialize_heart_service():
    """Initialize the heart service"""
    service = get_heart_service()
    return await service.initialize()