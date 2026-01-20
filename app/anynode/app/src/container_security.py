#!/usr/bin/env python3
"""
CogniKube Container Security - Layer 1
Wraps binary protocol in secure container
"""

from binary_security_layer import secure_comm
import asyncio
import socket
from typing import Dict

class SecureContainer:
    def __init__(self):
        self.binary_comm = secure_comm
        self.active_connections: Dict[str, socket.socket] = {}
        
    async def start_secure_server(self, service: str):
        """Start server on obscured port with binary protocol"""
        
        # Get obscured port for service
        port = self.binary_comm.ports.get_real_port(service)
        
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(5)
        
        print(f"🔒 Secure {service} server listening on obscured port {port}")
        
        while True:
            client_socket, addr = server_socket.accept()
            
            # Handle client in binary protocol
            asyncio.create_task(self.handle_binary_client(client_socket, service))
    
    async def handle_binary_client(self, client_socket: socket.socket, service: str):
        """Handle client using 13-bit binary protocol"""
        try:
            while True:
                # Receive binary data
                binary_data = client_socket.recv(1024)
                if not binary_data:
                    break
                
                # Decode using 13-bit protocol
                try:
                    port = self.binary_comm.ports.get_real_port(service)
                    decoded_message = self.binary_comm.secure_decode(binary_data, port, service)
                    
                    # Process message (placeholder)
                    response = f"Processed: {decoded_message}"
                    
                    # Encode response back to binary
                    binary_response, _ = self.binary_comm.secure_encode(response, service)
                    
                    # Send binary response
                    client_socket.send(binary_response)
                    
                except ValueError as e:
                    # Invalid port/service - close connection
                    print(f"Security violation: {e}")
                    break
                    
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client_socket.close()

# Container instance
container = SecureContainer()

# Start multiple secure services
async def start_all_services():
    """Start all secure services on obscured ports"""
    services = ["api", "ws", "health", "admin"]
    
    tasks = []
    for service in services:
        task = asyncio.create_task(container.start_secure_server(service))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print("🛡️ Starting CogniKube Secure Container...")
    print("📡 13-bit binary protocol active")
    print("🔀 Port obscuration enabled")
    
    asyncio.run(start_all_services())