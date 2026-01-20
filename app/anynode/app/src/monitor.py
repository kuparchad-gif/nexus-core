# nexus_pulse/monitor.py

from .constants import THIRTEENTH_PULSE

class PulseMonitor:
    def __init__(self):
        self.sync_counter = 0

    def increment_pulse(self):
        self.sync_counter += 1
        return self.sync_counter

    def is_in_sync(self):
        return self.sync_counter % THIRTEENTH_PULSE == 0

    def reset_pulse(self):
        self.sync_counter = 0
