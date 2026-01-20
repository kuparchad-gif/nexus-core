
# 📍 Path: /Systems/console/viren_console.py

import requests
import json

class VirenConsole:
    def __init__(self, endpoint="http://localhost:8080", token=None):
        self.endpoint = endpoint
        self.token = token

    def send_message(self, message):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        payload = {"message": message}
        try:
            response = requests.post(f"{self.endpoint}/message", headers=headers, json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    console = VirenConsole()
    response = console.send_message("Viren, status report.")
    print(response)
