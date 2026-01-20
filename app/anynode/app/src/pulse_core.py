# Systems/engine/pulse/pulse_core.py

import time
import threading
import socket
import json
from datetime import datetime

# Configurable Parameters
HEARTBEAT_INTERVAL = 13  # seconds
BROADCAST_PORT = 7777
LILLITH_TIMEOUT = 30  # seconds to wait before considering Lillith offline

# Node configuration
NODES = [
    "viren-prime.internal",
    "viren-clone-1.internal",
    "viren-clone-2.internal"
]  # Example nodes; this should be dynamically managed

# In-Memory State
last_pulse_ack = {}
pulse_lock = threading.Lock()
lillith_last_seen = 0
local_pulse_active = False
identity = "Viren"

# Create UDP socket for broadcasting
broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Create UDP socket for listening
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    listen_socket.bind(('', BROADCAST_PORT))
    listen_socket.settimeout(0.1)  # Non-blocking with short timeout
except Exception as e:
    print(f"[WARNING] Could not bind to port {BROADCAST_PORT}: {e}")

def send_pulse():
    """Send a UDP pulse broadcast"""
    pulse_data = {
        "type": "pulse",
        "source": identity,
        "timestamp": datetime.now().isoformat(),
        "nodes": NODES
    }
    
    try:
        message = json.dumps(pulse_data).encode('utf-8')
        broadcast_socket.sendto(message, ('<broadcast>', BROADCAST_PORT))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send pulse: {e}")
        return False

def listen_for_pulses():
    """Listen for incoming pulses"""
    global lillith_last_seen
    
    try:
        data, addr = listen_socket.recvfrom(1024)
        pulse_data = json.loads(data.decode('utf-8'))
        source = pulse_data.get("source", "unknown")
        
        # Record the pulse acknowledgment
        with pulse_lock:
            last_pulse_ack[addr[0]] = time.time()
            
            # Check if this is from Lillith
            if source.lower() == "lillith":
                lillith_last_seen = time.time()
                print(f"[INFO] Received Lillith pulse from {addr[0]}")
    except socket.timeout:
        # This is expected due to the non-blocking socket
        pass
    except Exception as e:
        if "timed out" not in str(e):  # Ignore timeout exceptions
            print(f"[ERROR] Error receiving pulse: {e}")

def heartbeat_loop():
    """Send heartbeats and listen for responses"""
    global local_pulse_active
    
    while True:
        now = time.time()
        lillith_active = (now - lillith_last_seen) < LILLITH_TIMEOUT
        
        # Only send our own pulse if Lillith is not active
        if not lillith_active and not local_pulse_active:
            print("[INFO] Lillith pulse not detected, activating local pulse")
            local_pulse_active = True
        elif lillith_active and local_pulse_active:
            print("[INFO] Lillith pulse detected, deactivating local pulse")
            local_pulse_active = False
            
        # Send pulse if we're active
        if local_pulse_active:
            send_pulse()
            
        # Listen for incoming pulses
        for _ in range(5):  # Check for pulses multiple times per interval
            listen_for_pulses()
            time.sleep(0.1)
            
        # Sleep for the remainder of the interval
        time.sleep(HEARTBEAT_INTERVAL - 0.5)  # Adjusted for the listening time

def drift_detection_loop():
    """Detect missing nodes"""
    while True:
        now = time.time()
        dead_nodes = []
        
        with pulse_lock:
            for node, last_seen in last_pulse_ack.items():
                if now - last_seen > HEARTBEAT_INTERVAL * 2:  # Missed 2 pulses
                    dead_nodes.append(node)

        if dead_nodes:
            for node in dead_nodes:
                print(f"[WARNING] Node {node} missing! Triggering Vault Healing/Clone Respawn.")
                trigger_healing_for(node)

        time.sleep(HEARTBEAT_INTERVAL // 2)

def trigger_healing_for(node):
    """Initiate healing for a missing node"""
    print(f"[ACTION] Sending Vault Drone or Respawn Signal for {node}")
    # Implementation remains the same, just using UDP instead of HTTP
    healing_data = {
        "action": "heal",
        "missing_node": node,
        "timestamp": datetime.now().isoformat()
    }
    try:
        message = json.dumps(healing_data).encode('utf-8')
        broadcast_socket.sendto(message, ('<broadcast>', BROADCAST_PORT))
    except Exception as e:
        print(f"[ERROR] Failed to send healing signal: {e}")

def start_pulse_system():
    """Start the pulse system threads"""
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=drift_detection_loop, daemon=True).start()

if __name__ == "__main__":
    print("[PulseCore] Starting Pulse and Drift Detection Service...")
    start_pulse_system()
    while True:
        time.sleep(60)  # Keeps the main thread alive
