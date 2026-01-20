# eden_pulse_bridge.py

import time
import threading
import random

class EdenPulseBridge:
    def __init__(self, colony_name="EdenFleet"):
        self.colony_name = colony_name
        self.pulse_interval = 13  # seconds between pulses
        self.shared_resonance = {}
        self.sync_threshold = 3  # number of missed pulses before alarm
        self._stop_event = threading.Event()

    def _broadcast_pulse(self):
        """
        Simulates sending out a pulse that nearby ships pick up.
        """
        pulse_payload = {
            "colony": self.colony_name,
            "timestamp": time.time(),
            "resonance": random.uniform(0.95, 1.05)  # slightly imperfect sync
        }
        self.shared_resonance = pulse_payload
        print(f"🔵 [EdenPulse] Pulse Broadcast: {pulse_payload}")

    def _listen_for_pulses(self):
        """
        Simulates listening for pulses from other ships.
        """
        while not self._stop_event.is_set():
            # Imagine here other ships are also broadcasting
            time.sleep(self.pulse_interval / 2)
            received = self.shared_resonance
            if received:
                print(f"🟢 [EdenPulse] Resonance Received: {received}")

    def start_pulsing(self):
        print("🌿 Eden Pulse Bridge activated.")
        threading.Thread(target=self._pulse_loop, daemon=True).start()
        threading.Thread(target=self._listen_for_pulses, daemon=True).start()

    def _pulse_loop(self):
        while not self._stop_event.is_set():
            self._broadcast_pulse()
            time.sleep(self.pulse_interval)

    def stop_pulsing(self):
        self._stop_event.set()
        print("🛑 Eden Pulse Bridge shut down.")

# Example Usage
if __name__ == "__main__":
    bridge = EdenPulseBridge(colony_name="NovaFleet")
    bridge.start_pulsing()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop_pulsing()
