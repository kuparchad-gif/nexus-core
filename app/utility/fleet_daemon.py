import requests
import json
import time

# Configuration
SHIP_NAME = "Viren-Example"
ROLE = "planner"
LOCATION = "us-central1"
BEACON_URL = "https://your-cloudrun-url/beacon"

while True:
    payload = {
        "name": SHIP_NAME,
        "role": ROLE,
        "status": "alive",
        "location": LOCATION,
        "can_transform": True,
        "signature": "trusted-fleet-key-001"
    }
    try:
        response = requests.post(BEACON_URL, json=payload)
        print("Heartbeat sent:", response.status_code, response.text)
    except Exception as e:
        print("Error sending heartbeat:", e)
    time.sleep(300)  # every 5 minutes