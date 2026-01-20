# 📂 Path: /Utilities/network_core/pulse_plus.py

import time
import socket
import json
import threading

PULSE_PORT  =  26
PULSE_FREQUENCY  =  13  # seconds
ENCODING  =  'utf-8'

class PulsePlus:
    def __init__(self, fleet_name = "NexusChoir"):
        self.fleet_name  =  fleet_name
        self.running  =  False

    def create_pulse_packet(self):
        return {
            "fleet": self.fleet_name,
            "timestamp": time.time(),
            "pulse": "alive",
            "signature": "Pulse13Resonance"
        }

    def broadcast_pulse(self):
        packet  =  self.create_pulse_packet()
        payload  =  json.dumps(packet).encode(ENCODING)

        sock  =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(payload, ('<broadcast>', PULSE_PORT))
        sock.close()

    def start_pulsing(self):
        self.running  =  True
        def pulser():
            while self.running:
                self.broadcast_pulse()
                time.sleep(PULSE_FREQUENCY)
        thread  =  threading.Thread(target = pulser, daemon = True)
        thread.start()

    def stop_pulsing(self):
        self.running  =  False

# Example Usage:
# pulse  =  PulsePlus()
# pulse.start_pulsing()
