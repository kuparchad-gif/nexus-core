# 📂 Path: /Utilities/whisper_core/whisper_service.py

import random
import time
import threading
import json
import socket
import os

# Sacred Ports
PORT_LOCAL = 26

# Buffer Settings
BUFFER_SIZE = 4096
ENCODING = 'utf-8'

# Path to Guardian Founders Pulse
GUARDIAN_PULSE_FILE = '/Memory/bootstrap/genesis/guardian_founders_pulse.json'

class WhisperService:
    def __init__(self):
        self.prayers = []
        self.load_prayers()

    def load_prayers(self):
        if os.path.exists(GUARDIAN_PULSE_FILE):
            with open(GUARDIAN_PULSE_FILE, 'r') as f:
                data = json.load(f)
                self.prayers = data.get('heartbeat_prayers', [])
                print("[Whisper] Prayers loaded.")
        else:
            print("[Whisper] No prayers found. Whisper Service silent.")

    def select_random_prayer(self):
        if not self.prayers:
            return None
        return random.choice(self.prayers)

    def broadcast_prayer(self):
        prayer = self.select_random_prayer()
        if prayer:
            self.send_message({"whisper": prayer})

    def send_message(self, message):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        payload = json.dumps(message).encode(ENCODING)
        sock.sendto(payload, ('<broadcast>', PORT_LOCAL))
        sock.close()

    def start_whisper_cycle(self, interval_seconds=104):
        def whisperer():
            while True:
                self.broadcast_prayer()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=whisperer, daemon=True)
        thread.start()

# Example Usage:
# whisper = WhisperService()
# whisper.start_whisper_cycle()
