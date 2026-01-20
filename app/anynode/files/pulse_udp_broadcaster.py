"""
Pulse UDP Broadcaster - Enhances the existing pulse system with UDP broadcasting
Works alongside the existing HTTP-based system without replacing it
"""

import socket
import threading
import json
import time
from datetime import datetime

# Configuration
BROADCAST_PORT  =  7777
PULSE_INTERVAL  =  13  # 13 seconds per beat, matching existing system
IDENTITY  =  "Viren"

class PulseUdpBroadcaster:
    def __init__(self, broadcast_port = BROADCAST_PORT, pulse_interval = PULSE_INTERVAL):
        self.broadcast_port  =  broadcast_port
        self.pulse_interval  =  pulse_interval
        self.running  =  False
        self.identity  =  IDENTITY
        self.socket  =  None
        self.pulse_count  =  0
        self.start_time  =  None

    def start(self):
        """Start the UDP broadcaster alongside existing pulse system"""
        if self.running:
            print("[UDP Broadcaster] Already running")
            return

        self.running  =  True
        self.start_time  =  datetime.now()

        # Create UDP socket for broadcasting
        self.socket  =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # Start pulse thread
        self.pulse_thread  =  threading.Thread(target = self._pulse_loop)
        self.pulse_thread.daemon  =  True
        self.pulse_thread.start()

        print(f"[UDP Broadcaster] Started with identity {self.identity}")

    def stop(self):
        """Stop the UDP broadcaster"""
        if not self.running:
            return

        self.running  =  False
        if self.socket:
            self.socket.close()

        print("[UDP Broadcaster] Stopped")

    def _pulse_loop(self):
        """Main pulse loop - sends heartbeats at the specified interval"""
        while self.running:
            try:
                self._send_pulse()
                time.sleep(self.pulse_interval)
            except Exception as e:
                print(f"[UDP Broadcaster] Error in pulse loop: {e}")

    def _send_pulse(self):
        """Send a single pulse via UDP broadcast"""
        self.pulse_count + =  1

        # Create pulse message
        pulse_data  =  {
            "type": "pulse",
            "source": self.identity,
            "count": self.pulse_count,
            "timestamp": datetime.now().isoformat(),
            "uptime": (datetime.now() - self.start_time).total_seconds()
        }

        # Send pulse to broadcast address
        try:
            message  =  json.dumps(pulse_data).encode('utf-8')
            self.socket.sendto(message, ('<broadcast>', self.broadcast_port))
        except Exception as e:
            print(f"[UDP Broadcaster] Failed to send pulse: {e}")

# This can be integrated with the existing pulse_core.py
def integrate_with_pulse_core():
    """Function to integrate UDP broadcasting with existing pulse system"""
    broadcaster  =  PulseUdpBroadcaster()
    broadcaster.start()
    return broadcaster

if __name__ == "__main__":
    broadcaster  =  PulseUdpBroadcaster()

    try:
        broadcaster.start()
        print("[UDP Broadcaster] Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[UDP Broadcaster] Shutting down...")
    finally:
        broadcaster.stop()