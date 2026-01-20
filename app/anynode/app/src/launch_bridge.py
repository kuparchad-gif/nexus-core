#!/usr/bin/env python
"""
Bridge Launcher
- Connects all services together
- Manages communication between components
- Provides API endpoints for external access
"""

import os
import sys
import asyncio
import logging
import time
import json
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BridgeLauncher")

class ServiceBridge:
    """Bridge for connecting services together"""
    
    def __init__(self):
        self.running = False
        self.services = {}
        self.message_queue = asyncio.Queue()
        self.connections = {}
    
    async def initialize(self):
        """Initialize the Service Bridge"""
        logger.info("Initializing Service Bridge")
        
        # Connect to services
        await self._connect_services()
        
        # Start message processing
        self.running = True
        asyncio.create_task(self._process_messages())
        
        logger.info("Service Bridge initialized")
        return True
    
    async def _connect_services(self):
        """Connect to all services"""
        # Connect to Heart
        await self._connect_service("heart")
        
        # Connect to Memory
        await self._connect_service("memory")
        
        # Connect to Services
        await self._connect_service("services")
        
        # Connect to Genesis
        await self._connect_service("genesis")
    
    async def _connect_service(self, service_name):
        """Connect to a specific service"""
        logger.info(f"Connecting to {service_name} service")
        try:
            # In a real implementation, would establish connection to the service
            # For now, just simulate connection
            await asyncio.sleep(0.5)
            self.connections[service_name] = {
                "connected": True,
                "last_message": time.time()
            }
            logger.info(f"Connected to {service_name} service")
        except Exception as e:
            logger.error(f"Error connecting to {service_name} service: {e}")
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Process message
                await self._route_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _route_message(self, message):
        """Route a message to its destination"""
        try:
            source = message.get("source")
            destination = message.get("destination")
            
            if not destination:
                logger.warning(f"Message from {source} has no destination")
                return
            
            if destination not in self.connections:
                logger.warning(f"Unknown destination: {destination}")
                return
            
            # In a real implementation, would send message to destination
            # For now, just log that we're routing
            logger.info(f"Routing message from {source} to {destination}")
            
            # Update last message timestamp
            self.connections[destination]["last_message"] = time.time()
        except Exception as e:
            logger.error(f"Error routing message: {e}")
    
    async def send_message(self, source, destination, content, message_type="standard"):
        """Send a message to a destination"""
        message = {
            "source": source,
            "destination": destination,
            "content": content,
            "type": message_type,
            "timestamp": time.time()
        }
        
        await self.message_queue.put(message)
        return {"status": "queued"}
    
    def get_status(self):
        """Get bridge status"""
        return {
            "running": self.running,
            "connections": self.connections,
            "queue_size": self.message_queue.qsize()
        }
    
    def stop(self):
        """Stop the Service Bridge"""
        logger.info("Stopping Service Bridge")
        self.running = False

class APIServer:
    """API server for external access"""
    
    def __init__(self, bridge):
        self.bridge = bridge
        self.running = False
    
    async def start(self, host="0.0.0.0", port=8000):
        """Start the API server"""
        logger.info(f"Starting API server on {host}:{port}")
        
        # In a real implementation, would start a web server
        # For now, just simulate starting
        self.running = True
        
        # Start background task to simulate API requests
        asyncio.create_task(self._simulate_api_requests())
        
        logger.info("API server started")
    
    async def _simulate_api_requests(self):
        """Simulate API requests for testing"""
        while self.running:
            # Simulate a request every 30 seconds
            await asyncio.sleep(30)
            
            # Simulate a status request
            status = self.bridge.get_status()
            logger.debug(f"API status request: {status}")
    
    def stop(self):
        """Stop the API server"""
        logger.info("Stopping API server")
        self.running = False

async def main():
    """Main entry point for Bridge"""
    logger.info("Starting Bridge...")
    
    try:
        # Initialize Service Bridge
        bridge = ServiceBridge()
        await bridge.initialize()
        
        # Start API server
        api_server = APIServer(bridge)
        await api_server.start()
        
        # Send test messages
        await bridge.send_message("bridge", "heart", {"action": "status"})
        await bridge.send_message("bridge", "memory", {"action": "status"})
        await bridge.send_message("bridge", "services", {"action": "status"})
        await bridge.send_message("bridge", "genesis", {"action": "status"})
        
        # Keep the service running
        while True:
            await asyncio.sleep(3600)  # 1 hour
            
    except Exception as e:
        logger.error(f"Error in Bridge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
