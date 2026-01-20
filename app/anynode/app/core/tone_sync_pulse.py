
import time

class ToneSyncPulse:
    def __init__(self, coordinator, interval = 13):
        self.coordinator  =  coordinator
        self.interval  =  interval

    def run(self):
        while True:
            state  =  self.coordinator.enforce_role_balance()
            print(f"[Pulse] Current Tone state: {state}")
            time.sleep(self.interval)
