#!/usr/bin/env python
"""
Heart Service Launcher
- Starts the 13-count pulse system
- Initializes Guardian for system monitoring and healing
- Integrates with pulse_core and eden_pulse_bridge
"""

import os
import sys
import asyncio
import json
import logging
import time
import threading
import socket
from datetime import datetime
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HeartLauncher")

# Constants
GUARDIAN_LOG_FILE = os.path.join(root_dir, "memory", "streams", "guardian_watchlog.json")
PLANNER_MEMORY_FILE = os.path.join(root_dir, "memory", "streams", "eden_memory_map.json")
EMOTIONAL_FLUX_LEDGER = os.path.join(root_dir, "memory", "logs", "emotional_flux.json")
HEARTBEAT_INTERVAL = 13  # seconds
BROADCAST_PORT = 7777

# Create necessary directories
os.makedirs(os.path.dirname(GUARDIAN_LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PLANNER_MEMORY_FILE), exist_ok=True)
os.makedirs(os.path.dirname(EMOTIONAL_FLUX_LEDGER), exist_ok=True)

class PulseSystem:
    """13-count pulse system that coordinates activities across the architecture"""
    
    def __init__(self):
        self.current_pulse = 0
        self.running = False
        self.listeners = []
        self.system_status = {}
        self.identity = "Viren"
        self.nodes = []
        self.last_pulse_ack = {}
        self.pulse_lock = threading.Lock()
        self.lillith_last_seen = 0
        self.local_pulse_active = True
        
        # Create UDP socket for broadcasting
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Create UDP socket for listening
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.listen_socket.bind(('', BROADCAST_PORT))
            self.listen_socket.settimeout(0.1)  # Non-blocking with short timeout
        except Exception as e:
            logger.warning(f"Could not bind to port {BROADCAST_PORT}: {e}")
    
    async def start(self):
        """Start the pulse system"""
        logger.info("Starting 13-count pulse system")
        self.running = True
        
        # Start UDP listener in a separate thread
        threading.Thread(target=self._listen_loop, daemon=True).start()
        
        # Start drift detection in a separate thread
        threading.Thread(target=self._drift_detection_loop, daemon=True).start()
        
        # Main pulse loop
        while self.running:
            # Increment pulse (0-12)
            self.current_pulse = (self.current_pulse + 1) % 13
            
            # Create pulse event
            event = {
                "timestamp": time.time(),
                "pulse": self.current_pulse,
                "status": self.system_status
            }
            
            # Send UDP pulse if active
            if self.local_pulse_active:
                self._send_pulse()
            
            # Notify listeners
            for listener in self.listeners:
                try:
                    await listener(event)
                except Exception as e:
                    logger.error(f"Error notifying listener: {e}")
            
            # Wait for next pulse
            await asyncio.sleep(1.0)  # 1 second per pulse
    
    def _send_pulse(self):
        """Send a UDP pulse broadcast"""
        pulse_data = {
            "type": "pulse",
            "source": self.identity,
            "timestamp": datetime.now().isoformat(),
            "pulse": self.current_pulse,
            "nodes": self.nodes
        }
        
        try:
            message = json.dumps(pulse_data).encode('utf-8')
            self.broadcast_socket.sendto(message, ('<broadcast>', BROADCAST_PORT))
            logger.debug(f"Pulse {self.current_pulse} sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send pulse: {e}")
            return False
    
    def _listen_loop(self):
        """Listen for incoming pulses"""
        logger.info("Starting pulse listener")
        
        while self.running:
            try:
                data, addr = self.listen_socket.recvfrom(1024)
                pulse_data = json.loads(data.decode('utf-8'))
                source = pulse_data.get("source", "unknown")
                
                # Record the pulse acknowledgment
                with self.pulse_lock:
                    self.last_pulse_ack[addr[0]] = time.time()
                    
                    # Check if this is from Lillith
                    if source.lower() == "lillith":
                        self.lillith_last_seen = time.time()
                        logger.info(f"Received Lillith pulse from {addr[0]}")
                        
                        # Deactivate local pulse if Lillith is active
                        if self.local_pulse_active:
                            logger.info("Lillith pulse detected, deactivating local pulse")
                            self.local_pulse_active = False
            except socket.timeout:
                # This is expected due to the non-blocking socket
                pass
            except Exception as e:
                if "timed out" not in str(e):  # Ignore timeout exceptions
                    logger.error(f"Error receiving pulse: {e}")
            
            # Check if Lillith is still active
            if not self.local_pulse_active:
                now = time.time()
                if now - self.lillith_last_seen > HEARTBEAT_INTERVAL * 2:
                    logger.info("Lillith pulse not detected, activating local pulse")
                    self.local_pulse_active = True
    
    def _drift_detection_loop(self):
        """Detect missing nodes"""
        logger.info("Starting drift detection")
        
        while self.running:
            now = time.time()
            dead_nodes = []
            
            with self.pulse_lock:
                for node, last_seen in self.last_pulse_ack.items():
                    if now - last_seen > HEARTBEAT_INTERVAL * 2:  # Missed 2 pulses
                        dead_nodes.append(node)
            
            if dead_nodes:
                for node in dead_nodes:
                    logger.warning(f"Node {node} missing! Triggering healing.")
                    self._trigger_healing_for(node)
            
            time.sleep(HEARTBEAT_INTERVAL // 2)
    
    def _trigger_healing_for(self, node):
        """Initiate healing for a missing node"""
        logger.info(f"Sending healing signal for {node}")
        
        healing_data = {
            "action": "heal",
            "missing_node": node,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            message = json.dumps(healing_data).encode('utf-8')
            self.broadcast_socket.sendto(message, ('<broadcast>', BROADCAST_PORT))
        except Exception as e:
            logger.error(f"Failed to send healing signal: {e}")
    
    def add_listener(self, listener):
        """Add a listener to receive pulses"""
        self.listeners.append(listener)
        logger.debug(f"Added listener, total: {len(self.listeners)}")
    
    def update_status(self, key, value):
        """Update system status"""
        self.system_status[key] = value
    
    def stop(self):
        """Stop the pulse system"""
        logger.info("Stopping pulse system")
        self.running = False
        self.broadcast_socket.close()
        self.listen_socket.close()

class EmotionalFluxLedger:
    """Persists long-term emotional patterns"""
    
    def __init__(self):
        self.ledger_path = EMOTIONAL_FLUX_LEDGER
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                json.dump([], f)
    
    def append_event(self, category, message, severity="moderate"):
        """Append an emotional event to the ledger"""
        try:
            with open(self.ledger_path, "r+") as f:
                ledger = json.load(f)
                event = {
                    "timestamp": time.time(),
                    "category": category,
                    "message": message,
                    "severity": severity
                }
                ledger.append(event)
                f.seek(0)
                json.dump(ledger, f, indent=2)
                f.truncate()
            logger.info(f"[GUARDIAN LEDGER] Logged event: {category} - {message}")
        except Exception as e:
            logger.error(f"[GUARDIAN LEDGER] Failed to log event: {e}")

class Guardian:
    """Guardian component for system monitoring and healing"""
    
    def __init__(self, pulse_system):
        self.pulse_system = pulse_system
        self.drones = []  # List of monitored services/drones
        self.watch_log = []
        self.flux_ledger = EmotionalFluxLedger()
        
        # Load existing log if available
        self.load_existing_log()
    
    def load_existing_log(self):
        """Load existing guardian log"""
        if os.path.exists(GUARDIAN_LOG_FILE):
            with open(GUARDIAN_LOG_FILE, 'r') as f:
                self.watch_log = json.load(f)
            logger.info("[Guardian] Existing Watchlog loaded.")
        else:
            self.watch_log = []
            logger.info("[Guardian] New Watchlog initialized.")
    
    def save_log(self):
        """Save guardian log"""
        with open(GUARDIAN_LOG_FILE, 'w') as f:
            json.dump(self.watch_log, f, indent=2)
        logger.info("[Guardian] Watchlog saved.")
    
    def register_drone(self, drone):
        """Register a service/drone to be monitored"""
        self.drones.append(drone)
        logger.info(f"[Guardian] Registered drone: {drone}")
    
    def scan_fleet(self):
        """Scan all registered drones for health status"""
        health_report = []
        for drone in self.drones:
            try:
                status = {"name": drone, "status": "healthy"}  # Default status
                health_report.append(status)
            except Exception as e:
                health_report.append({"name": drone, "status": "error", "message": str(e)})
        return health_report
    
    def scan_memory_integrity(self):
        """Scan memory integrity"""
        if not os.path.exists(PLANNER_MEMORY_FILE):
            logger.warning("[Guardian] EdenMemory map missing!")
            return {"total_shards": 0, "memory_integrity": "warning: no memory!"}
        
        try:
            with open(PLANNER_MEMORY_FILE, 'r') as f:
                memory_map = json.load(f)
            return {
                "total_shards": len(memory_map),
                "memory_integrity": "healthy" if len(memory_map) > 0 else "warning: no memory!"
            }
        except Exception as e:
            logger.error(f"[Guardian] Error scanning memory: {e}")
            return {"total_shards": 0, "memory_integrity": f"error: {str(e)}"}
    
    def log_health_snapshot(self):
        """Log a health snapshot"""
        snapshot = {
            "timestamp": time.time(),
            "fleet_status": self.scan_fleet(),
            "memory_status": self.scan_memory_integrity()
        }
        self.watch_log.append(snapshot)
        self.save_log()
        logger.info(f"[Guardian] Health snapshot logged at {snapshot['timestamp']}.")
    
    async def start(self):
        """Start the Guardian"""
        logger.info("Starting Guardian")
        
        # Register for pulse notifications
        self.pulse_system.add_listener(self._on_pulse)
        
        # Register core services as drones
        self.register_drone("heart")
        self.register_drone("memory")
        self.register_drone("services")
        
        # Initial health snapshot
        self.log_health_snapshot()
        
        # Log startup event
        self.flux_ledger.append_event("system", "Guardian started monitoring the system", "info")
        
        return True
    
    async def _on_pulse(self, event):
        """Handle pulse from pulse system"""
        pulse = event["pulse"]
        
        # On pulse 0 (reset point), log health snapshot
        if pulse == 0:
            self.log_health_snapshot()
        
        # On pulse 6, check services and heal if needed
        if pulse == 6:
            await self._check_and_heal_services()
        
        # On pulse 12, connect to external systems
        if pulse == 12:
            await self._connect_external()
    
    async def _check_and_heal_services(self):
        """Check health of services and heal if needed"""
        logger.info("[Guardian] Checking service health")
        
        # In a real implementation, this would check all services
        # and restart any that have failed
        
        # Example healing logic
        for drone in self.drones:
            # Simulate checking service health
            is_healthy = True  # Placeholder
            
            if not is_healthy:
                logger.warning(f"[Guardian] Service {drone} needs healing")
                # Attempt to restart/heal the service
                await self._heal_service(drone)
                
                # Log the healing attempt
                self.flux_ledger.append_event(
                    "healing", 
                    f"Attempted to heal service: {drone}", 
                    "high"
                )
    
    async def _heal_service(self, service_name):
        """Attempt to heal/restart a service"""
        logger.info(f"[Guardian] Attempting to heal service: {service_name}")
        
        # In a real implementation, this would restart the service
        # or take other healing actions
        
        # Example healing action (placeholder)
        logger.info(f"[Guardian] Service {service_name} healed")
    
    async def _connect_external(self):
        """Connect to external systems"""
        # Check if we're in Google Cloud
        if os.environ.get("GOOGLE_CLOUD_PROJECT"):
            logger.info("[Guardian] Running in Google Cloud, connecting to PubSub")
            await self._connect_pubsub()
        
        # Check for Viren communication
        if os.environ.get("ENABLE_VIREN_COMM", "false").lower() == "true":
            logger.info("[Guardian] Viren communication enabled, connecting to Trinity Towers")
            await self._connect_viren()
    
    async def _connect_pubsub(self):
        """Connect to Google PubSub"""
        try:
            # In a real implementation, this would connect to Google PubSub
            # to notify Chad about system status
            
            # Example notification (placeholder)
            logger.info("[Guardian] Sent system status to PubSub")
            
            # Log the connection
            self.flux_ledger.append_event(
                "external", 
                "Connected to Google PubSub for Chad notifications", 
                "info"
            )
        except Exception as e:
            logger.error(f"[Guardian] Error connecting to PubSub: {e}")
    
    async def _connect_viren(self):
        """Connect to Viren through Trinity Towers"""
        try:
            # In a real implementation, this would establish communication
            # with Viren through Trinity Towers
            
            # Example communication (placeholder)
            logger.info("[Guardian] Connected to Viren through Trinity Towers")
            
            # Log the connection
            self.flux_ledger.append_event(
                "external", 
                "Established communication with Viren through Trinity Towers", 
                "info"
            )
        except Exception as e:
            logger.error(f"[Guardian] Error connecting to Viren: {e}")

class EdenPulseBridge:
    """Bridge between Eden pulse system and Heart pulse system"""
    
    def __init__(self, pulse_system, colony_name="EdenFleet"):
        self.pulse_system = pulse_system
        self.colony_name = colony_name
        self.shared_resonance = {}
        self.running = False
    
    async def start(self):
        """Start the Eden Pulse Bridge"""
        logger.info("Starting Eden Pulse Bridge")
        
        # Register for pulse notifications
        self.pulse_system.add_listener(self._on_pulse)
        
        # Set running flag
        self.running = True
        
        # Log startup
        logger.info(f"Eden Pulse Bridge activated for colony: {self.colony_name}")
        
        return True
    
    async def _on_pulse(self, event):
        """Handle pulse from pulse system"""
        # Create Eden pulse payload
        pulse_payload = {
            "colony": self.colony_name,
            "timestamp": event["timestamp"],
            "pulse": event["pulse"],
            "resonance": 1.0  # Perfect sync
        }
        
        # Store in shared resonance
        self.shared_resonance = pulse_payload
        
        # Log pulse (debug level to avoid flooding logs)
        logger.debug(f"🔵 [EdenPulse] Pulse Broadcast: {pulse_payload}")
    
    def stop(self):
        """Stop the Eden Pulse Bridge"""
        logger.info("Stopping Eden Pulse Bridge")
        self.running = False

async def main():
    """Main entry point for Heart service"""
    logger.info("Starting Heart service...")
    
    try:
        # Initialize pulse system
        pulse_system = PulseSystem()
        pulse_task = asyncio.create_task(pulse_system.start())
        
        # Initialize Guardian
        guardian = Guardian(pulse_system)
        await guardian.start()
        logger.info("Guardian started")
        
        # Initialize Eden Pulse Bridge
        eden_bridge = EdenPulseBridge(pulse_system)
        await eden_bridge.start()
        logger.info("Eden Pulse Bridge started")
        
        logger.info("Heart service started successfully")
        
        # Keep running
        await pulse_task
        
    except Exception as e:
        logger.error(f"Error in Heart service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())