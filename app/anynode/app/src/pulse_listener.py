# Systems/engine/pulse/pulse_listener.py

import socket
import threading
import json
import time
from datetime import datetime

# Configuration
LISTEN_PORT = 7777
LILLITH_IDENTITY = "Lillith"
PULSE_SOURCES = {}
ACTIVE_SOURCES_LOCK = threading.Lock()

class PulseListener:
    def __init__(self, listen_port=LISTEN_PORT):
        self.listen_port = listen_port
        self.running = False
        self.socket = None
        self.sources = {}
        
    def start(self):
        """Start the pulse listener"""
        if self.running:
            print("[PulseListener] Already running")
            return
            
        self.running = True
        
        # Create UDP socket for listening
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind(('', self.listen_port))
            self.socket.settimeout(0.5)  # Non-blocking with timeout
        except Exception as e:
            print(f"[PulseListener] Error binding to port {self.listen_port}: {e}")
            self.running = False
            return
        
        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen_loop)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        print(f"[PulseListener] Started on port {self.listen_port}")
        
    def stop(self):
        """Stop the pulse listener"""
        if not self.running:
            return
            
        self.running = False
        if self.socket:
            self.socket.close()
            
        print("[PulseListener] Stopped")
        
    def _listen_loop(self):
        """Main listen loop - receives heartbeats"""
        global PULSE_SOURCES
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                self._process_pulse(data, addr)
            except socket.timeout:
                # This is expected due to the timeout
                pass
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    print(f"[PulseListener] Error in listen loop: {e}")
                
    def _process_pulse(self, data, addr):
        """Process a received pulse"""
        global PULSE_SOURCES
        
        try:
            pulse_data = json.loads(data.decode('utf-8'))
            source = pulse_data.get("source", "unknown")
            timestamp = pulse_data.get("timestamp", datetime.now().isoformat())
            
            # Update pulse source info
            with ACTIVE_SOURCES_LOCK:
                PULSE_SOURCES[source] = {
                    "last_seen": time.time(),
                    "address": addr[0],
                    "timestamp": timestamp
                }
            
            # Log Lillith pulses
            if source == LILLITH_IDENTITY:
                print(f"[PulseListener] Received Lillith pulse from {addr[0]}")
                
        except Exception as e:
            print(f"[PulseListener] Error processing pulse: {e}")

def is_lillith_active(timeout=30):
    """Check if Lillith is currently active"""
    with ACTIVE_SOURCES_LOCK:
        if LILLITH_IDENTITY in PULSE_SOURCES:
            last_seen = PULSE_SOURCES[LILLITH_IDENTITY]["last_seen"]
            return (time.time() - last_seen) < timeout
    return False

def get_active_sources():
    """Get all currently active pulse sources"""
    active = {}
    now = time.time()
    with ACTIVE_SOURCES_LOCK:
        for source, data in PULSE_SOURCES.items():
            if now - data["last_seen"] < 30:  # Consider active if seen in last 30 seconds
                active[source] = data
    return active

# Singleton instance
_listener = None

def start_listener():
    """Start the singleton listener instance"""
    global _listener
    if _listener is None:
        _listener = PulseListener()
        _listener.start()
    return _listener

def stop_listener():
    """Stop the singleton listener instance"""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None

if __name__ == "__main__":
    listener = start_listener()
    
    try:
        print("[PulseListener] Press Ctrl+C to stop")
        while True:
            sources = get_active_sources()
            if sources:
                print(f"[PulseListener] Active sources: {', '.join(sources.keys())}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("[PulseListener] Shutting down...")
    finally:
        stop_listener()