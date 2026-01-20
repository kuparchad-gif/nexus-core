# Systems/packet_drift.py

import random
import time

class PacketDrift:
    @staticmethod
    def calculate_drift(base_interval=5.0):
        """Returns a slightly modified heartbeat interval."""
        drift = random.uniform(-0.25, 0.25)  # +/- 5% drift
        new_interval = max(1.0, base_interval + drift)
        return round(new_interval, 2)

if __name__ == "__main__":
    base = 5.0
    for _ in range(10):
        print(f"Adjusted heartbeat interval: {PacketDrift.calculate_drift(base)} seconds")
        time.sleep(1)
