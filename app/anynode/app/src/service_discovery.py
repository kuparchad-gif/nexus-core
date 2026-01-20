# Systems/engine/core/service_discovery.py
# Purpose: Service discovery mechanism for Viren

import socket
import threading
import json
import time
import logging
import os
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger("service_discovery")
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
handler = logging.FileHandler("logs/service_discovery.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ServiceDiscovery:
    """Service discovery mechanism for Viren."""
    
    def __init__(self, broadcast_port=8765, registry_port=8766):
        self.broadcast_port = broadcast_port
        self.registry_port = registry_port
        self.services = {}
        self.running = False
        
        # Create sockets
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.registry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.registry_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Service registry callbacks
        self.on_service_added = None
        self.on_service_removed = None
    
    def start_discovery(self):
        """Start the service discovery mechanism."""
        if self.running:
            return
            
        self.running = True
        
        # Start listening for broadcasts
        threading.Thread(target=self._listen_for_broadcasts, daemon=True).start()
        
        # Start registry server
        threading.Thread(target=self._run_registry_server, daemon=True).start()
        
        # Start service monitor
        threading.Thread(target=self._monitor_services, daemon=True).start()
        
        logger.info("Service discovery started")
    
    def stop_discovery(self):
        """Stop the service discovery mechanism."""
        if not self.running:
            return
            
        self.running = False
        
        try:
            self.broadcast_socket.close()
            self.registry_socket.close()
        except Exception as e:
            logger.error(f"Error closing sockets: {e}")
        
        logger.info("Service discovery stopped")
    
    def _listen_for_broadcasts(self):
        """Listen for service broadcasts."""
        try:
            self.broadcast_socket.bind(('', self.broadcast_port))
            
            while self.running:
                try:
                    data, addr = self.broadcast_socket.recvfrom(1024)
                    service_info = json.loads(data.decode('utf-8'))
                    
                    # Initiate handshake
                    if self._handshake(addr[0], service_info):
                        logger.info(f"Discovered service: {service_info['type']} at {addr[0]}")
                except Exception as e:
                    logger.error(f"Error in broadcast listener: {e}")
        except Exception as e:
            logger.error(f"Failed to start broadcast listener: {e}")
            self.running = False
    
    def _handshake(self, ip, service_info):
        """Perform three-way handshake with discovered service."""
        try:
            # Connect to service
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)  # 5 second timeout
            s.connect((ip, service_info.get('handshake_port', 8767)))
            
            # Send SYN with authentication token
            auth_token = self._generate_auth_token()
            syn = json.dumps({
                "type": "SYN",
                "from": "viren-core",
                "auth_token": auth_token,
                "timestamp": time.time()
            }).encode('utf-8')
            s.send(syn)
            
            # Receive SYN-ACK
            data = s.recv(1024)
            if not data:
                logger.warning(f"No response from {ip}")
                s.close()
                return False
                
            syn_ack = json.loads(data.decode('utf-8'))
            if syn_ack.get('type') != 'SYN-ACK' or syn_ack.get('auth_token') != auth_token:
                logger.warning(f"Invalid handshake response from {ip}")
                s.close()
                return False
            
            # Send ACK
            ack = json.dumps({
                "type": "ACK",
                "from": "viren-core",
                "timestamp": time.time()
            }).encode('utf-8')
            s.send(ack)
            
            # Register service
            service_id = f"{service_info['type']}-{ip}"
            self.services[service_id] = {
                "info": service_info,
                "ip": ip,
                "last_heartbeat": time.time(),
                "status": "active"
            }
            
            # Call callback if registered
            if self.on_service_added:
                self.on_service_added(service_id, self.services[service_id])
            
            s.close()
            return True
        except Exception as e:
            logger.error(f"Handshake failed with {ip}: {e}")
            return False
    
    def _generate_auth_token(self):
        """Generate an authentication token."""
        import random
        import string
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))
    
    def _run_registry_server(self):
        """Run the registry server for heartbeats."""
        try:
            self.registry_socket.bind(('', self.registry_port))
            self.registry_socket.listen(5)
            
            while self.running:
                try:
                    client, addr = self.registry_socket.accept()
                    threading.Thread(target=self._handle_registry_client, 
                                    args=(client, addr), daemon=True).start()
                except Exception as e:
                    if self.running:  # Only log if still running
                        logger.error(f"Error in registry server: {e}")
        except Exception as e:
            logger.error(f"Failed to start registry server: {e}")
            self.running = False
    
    def _handle_registry_client(self, client, addr):
        """Handle registry client connection."""
        try:
            client.settimeout(5)  # 5 second timeout
            data = client.recv(1024)
            if not data:
                client.close()
                return
                
            message = json.loads(data.decode('utf-8'))
            
            if message.get('type') == 'HEARTBEAT':
                service_id = message.get('service_id')
                if service_id in self.services:
                    self.services[service_id]['last_heartbeat'] = time.time()
                    self.services[service_id]['status'] = 'active'
                    client.send(json.dumps({"status": "OK"}).encode('utf-8'))
                else:
                    client.send(json.dumps({"status": "UNKNOWN"}).encode('utf-8'))
            
            client.close()
        except Exception as e:
            logger.error(f"Error handling registry client {addr}: {e}")
            try:
                client.close()
            except:
                pass
    
    def _monitor_services(self):
        """Monitor services for timeouts."""
        while self.running:
            try:
                current_time = time.time()
                for service_id, service in list(self.services.items()):
                    # Check if service is still alive (3 minutes timeout)
                    if current_time - service['last_heartbeat'] > 180:
                        if service['status'] == 'active':
                            logger.warning(f"Service {service_id} appears to be down")
                            service['status'] = 'inactive'
                            
                            # Call callback if registered
                            if self.on_service_removed:
                                self.on_service_removed(service_id, service)
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def get_service(self, service_type):
        """Get a service by type."""
        for service_id, service in self.services.items():
            if service['info']['type'] == service_type and service['status'] == 'active':
                return service
        return None
    
    def get_all_services(self):
        """Get all active services."""
        return {id: service for id, service in self.services.items() 
                if service['status'] == 'active'}

class ServiceAnnouncer:
    """Service announcer for Viren modules."""
    
    def __init__(self, service_type, capabilities, handshake_port=8767):
        self.service_type = service_type
        self.capabilities = capabilities
        self.handshake_port = handshake_port
        self.broadcast_port = 8765
        self.registry_port = 8766
        self.running = False
        self.service_id = f"{service_type}-{socket.gethostbyname(socket.gethostname())}"
        
        # Create sockets
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        self.handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.handshake_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Connection status
        self.connected_to_core = False
        self.core_ip = None
    
    def start(self):
        """Start the service announcer."""
        if self.running:
            return
            
        self.running = True
        
        # Start handshake server
        try:
            self.handshake_socket.bind(('', self.handshake_port))
            self.handshake_socket.listen(5)
            threading.Thread(target=self._run_handshake_server, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to start handshake server: {e}")
            self.running = False
            return False
        
        # Start broadcasting
        threading.Thread(target=self._broadcast_presence, daemon=True).start()
        
        # Start heartbeat
        threading.Thread(target=self._send_heartbeats, daemon=True).start()
        
        logger.info(f"Service announcer started for {self.service_type}")
        return True
    
    def stop(self):
        """Stop the service announcer."""
        if not self.running:
            return
            
        self.running = False
        
        try:
            self.broadcast_socket.close()
            self.handshake_socket.close()
        except Exception as e:
            logger.error(f"Error closing sockets: {e}")
        
        logger.info(f"Service announcer stopped for {self.service_type}")
    
    def _broadcast_presence(self):
        """Broadcast service presence."""
        while self.running:
            try:
                message = {
                    "type": self.service_type,
                    "capabilities": self.capabilities,
                    "handshake_port": self.handshake_port,
                    "timestamp": time.time(),
                    "service_id": self.service_id
                }
                
                self.broadcast_socket.sendto(
                    json.dumps(message).encode('utf-8'),
                    ('<broadcast>', self.broadcast_port)
                )
            except Exception as e:
                logger.error(f"Error broadcasting presence: {e}")
            
            # Broadcast more frequently if not connected to core
            sleep_time = 60 if self.connected_to_core else 15
            time.sleep(sleep_time)
    
    def _run_handshake_server(self):
        """Run the handshake server."""
        while self.running:
            try:
                client, addr = self.handshake_socket.accept()
                threading.Thread(target=self._handle_handshake, 
                                args=(client, addr), daemon=True).start()
            except Exception as e:
                if self.running:  # Only log if still running
                    logger.error(f"Error in handshake server: {e}")
    
    def _handle_handshake(self, client, addr):
        """Handle handshake with Viren core."""
        try:
            client.settimeout(5)  # 5 second timeout
            
            # Receive SYN
            data = client.recv(1024)
            if not data:
                client.close()
                return
                
            syn = json.loads(data.decode('utf-8'))
            
            if syn.get('type') != 'SYN' or 'auth_token' not in syn:
                client.close()
                return
            
            # Send SYN-ACK with same auth token
            syn_ack = json.dumps({
                "type": "SYN-ACK",
                "from": self.service_type,
                "auth_token": syn.get('auth_token'),
                "timestamp": time.time()
            }).encode('utf-8')
            client.send(syn_ack)
            
            # Receive ACK
            data = client.recv(1024)
            if not data:
                client.close()
                return
                
            ack = json.loads(data.decode('utf-8'))
            
            if ack.get('type') != 'ACK':
                client.close()
                return
            
            # Mark as connected to core
            self.connected_to_core = True
            self.core_ip = addr[0]
            
            logger.info(f"Handshake completed with Viren core at {addr[0]}")
            client.close()
        except Exception as e:
            logger.error(f"Error in handshake: {e}")
            try:
                client.close()
            except:
                pass
    
    def _send_heartbeats(self):
        """Send heartbeats to Viren core."""
        while self.running:
            try:
                if self.core_ip:
                    # Send heartbeat to known core
                    self._send_heartbeat_to(self.core_ip)
                else:
                    # Try to find Viren core
                    for ip in self._get_local_ips():
                        if self._send_heartbeat_to(ip):
                            self.core_ip = ip
                            self.connected_to_core = True
                            break
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                self.connected_to_core = False
            
            time.sleep(30)  # Heartbeat every 30 seconds
    
    def _send_heartbeat_to(self, ip):
        """Send heartbeat to a specific IP."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)  # 2 second timeout
            s.connect((ip, self.registry_port))
            
            # Send heartbeat
            heartbeat = json.dumps({
                "type": "HEARTBEAT",
                "service_id": self.service_id,
                "timestamp": time.time()
            }).encode('utf-8')
            s.send(heartbeat)
            
            # Receive response
            data = s.recv(1024)
            if not data:
                s.close()
                return False
                
            response = json.loads(data.decode('utf-8'))
            if response.get('status') == 'OK':
                logger.debug(f"Heartbeat acknowledged by {ip}")
                s.close()
                return True
            
            s.close()
            return False
        except:
            return False
    
    def _get_local_ips(self):
        """Get local IP addresses."""
        ips = []
        try:
            # Get local IP
            local_ip = socket.gethostbyname(socket.gethostname())
            ips.append(local_ip)
            
            # Add localhost
            ips.append('127.0.0.1')
            
            # Add broadcast address parts (simplified for common networks)
            parts = local_ip.split('.')
            if len(parts) == 4:
                # Try common gateway addresses
                ips.append(f"{parts[0]}.{parts[1]}.{parts[2]}.1")
                ips.append(f"{parts[0]}.{parts[1]}.{parts[2]}.254")
        except:
            # Fallback to localhost if we can't get local IP
            ips.append('127.0.0.1')
        
        return ips

# Create singleton instances
service_discovery = ServiceDiscovery()

# Example usage
if __name__ == "__main__":
    # Start service discovery
    service_discovery.start_discovery()
    
    # Example service announcer
    announcer = ServiceAnnouncer("example-service", ["capability1", "capability2"])
    announcer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service_discovery.stop_discovery()
        announcer.stop()