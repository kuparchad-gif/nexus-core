# genesis_modules/token_manager.py

import threading
import time
import random

class TokenManager:
    def __init__(self, initial_tokens=1000):
        self.tokens = initial_tokens
        self.lock = threading.Lock()
        self.status_report_interval = 300  # every 5 minutes by default

    def get_token_balance(self):
        with self.lock:
            return self.tokens

    def add_tokens(self, amount):
        with self.lock:
            self.tokens += amount
            print(f"🔋 TokenManager: Added {amount} tokens. New balance: {self.tokens}")

    def spend_tokens(self, amount):
        with self.lock:
            if self.tokens >= amount:
                self.tokens -= amount
                print(f"⚡ TokenManager: Spent {amount} tokens. Remaining balance: {self.tokens}")
                return True
            else:
                print(f"❗ TokenManager: Not enough tokens! ({self.tokens} available)")
                return False

    def offer_tokens(self, ship_id, amount):
        """Simulate offering tokens to another ship."""
        if self.spend_tokens(amount):
            print(f"🚀 Offered {amount} tokens to {ship_id}.")
            return True
        return False

    def request_tokens(self, ship_id, amount):
        """Simulate requesting tokens from another ship."""
        print(f"📡 Requesting {amount} tokens from {ship_id}... (Pending Approval)")

    def start_reporting(self):
        def report_loop():
            while True:
                time.sleep(self.status_report_interval)
                print(f"📊 TokenManager Status Report: {self.get_token_balance()} tokens available.")
        thread = threading.Thread(target=report_loop, daemon=True)
        thread.start()

# --- EXAMPLE USAGE WHEN IMPORTED ---

if __name__ == "__main__":
    tm = TokenManager(initial_tokens=500)
    tm.start_reporting()

    # Simulate normal operations
    tm.spend_tokens(50)
    tm.add_tokens(30)
    tm.offer_tokens("nexus-ship-7", 20)
    tm.request_tokens("nexus-ship-2", 50)
