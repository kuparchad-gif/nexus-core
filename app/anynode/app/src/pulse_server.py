"""
Pulse Server - Sends heartbeat pulses at 13 beats per second
Acts as a fallback when Lillith's heartbeat is not detected
"""

import time
import socket
import threading
import json
import uuid
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pulse_server.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PulseServer")

# Configuration
PULSE_RATE = 1/13  # 13 beats per second
BROADCAST_PORT = 7777
IDENTITY = str(uuid.uuid4())
SERVER_NAME = "VironPulse"

class PulseServer:
    def __init__(self, broadcast_port=BROADCAST_PORT, pulse_rate=PULSE_RATE):
        self.broadcast_port = broadcast_port
        self.pulse_rate = pulse_rate
        self.running = False
        self.identity = IDENTITY
        self.server_name = SERVER_NAME
        self.socket = None
        self.pulse_count = 0
        self.start_time = None
        
    def start(self):
        """Start the pulse server"""
        if self.running:
            logger.warning("Pulse server already running")
            return
            
        self.running = True
        self.start_time = datetime.now()
        
        # Create UDP socket for broadcasting
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Start pulse thread
        self.pulse_thread = threading.Thread(target=self._pulse_loop)
        self.pulse_thread.daemon = True
        self.pulse_thread.start()
        
        logger.info(f"Pulse server started with identity {self.identity}")
        
    def stop(self):
        """Stop the pulse server"""
        if not self.running:
            return
            
        self.running = False
        if self.socket:
            self.socket.close()
            
        logger.info("Pulse server stopped")
        
    def _pulse_loop(self):
        """Main pulse loop - sends heartbeats at the specified rate"""
        while self.running:
            try:
                self._send_pulse()
                time.sleep(self.pulse_rate)
            except Exception as e:
                logger.error(f"Error in pulse loop: {e}")
                
    def _send_pulse(self):
        """Send a single pulse"""
        self.pulse_count += 1
        
        # Reset counter after 13 pulses
        if self.pulse_count > 13:
            self.pulse_count = 1
            
        # Create pulse message
        pulse_data = {
            "type": "pulse",
            "source": self.server_name,
            "identity": self.identity,
            "count": self.pulse_count,
            "timestamp": datetime.now().isoformat(),
            "uptime": (datetime.now() - self.start_time).total_seconds()
        }
        
        # Send pulse to broadcast address
        try:
            message = json.dumps(pulse_data).encode('utf-8')
            self.socket.sendto(message, ('<broadcast>', self.broadcast_port))
        except Exception as e:
            logger.error(f"Failed to send pulse: {e}")

def main():
    """Run the pulse server"""
    server = PulseServer()
    
    try:
        server.start()
        logger.info("Press Ctrl+C to stop the server")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down pulse server...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()