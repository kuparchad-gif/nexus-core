# 📂 Path: Systems/engine/comms/orc_smart_router.py

import requests, random, time, json
from cryptography.fernet import Fernet
from websocket_bridge import EdenWebSocketBridge

# Load your encryption key
FERNET_KEY = b'your-32-byte-base64-key=='  # Replace with secure fetch from Vault
cipher = Fernet(FERNET_KEY)

# List of all Lilith endpoints (this can rotate with DNS or be static)
LILITH_NODES = [
    "https://viren1.nexus-core.app",
    "https://viren2.nexus-core.app",
    "https://viren3.nexus-core.app"
]

# Example health scores or simulated loads
node_health = {url: 1.0 for url in LILITH_NODES}  # 1.0 = perfect, 0.0 = offline

# WebSocket Relay Backup
bridge = EdenWebSocketBridge(port=8765)
bridge.start_in_background()

def select_best_node():
    sorted_nodes = sorted(LILITH_NODES, key=lambda url: node_health.get(url, 0), reverse=True)
    return sorted_nodes[0]

def encrypt_payload(data: dict) -> str:
    return cipher.encrypt(json.dumps(data).encode()).decode()

def decrypt_payload(token: str) -> dict:
    return json.loads(cipher.decrypt(token.encode()).decode())

def forward_request(data: dict):
    try:
        best_node = select_best_node()
        encrypted = encrypt_payload(data)
        response = requests.post(f"{best_node}/relay", json={"payload": encrypted}, timeout=5)
        return response.json()
    except Exception as ex:
        print(f"🌐 Primary route failed: {ex}")
        asyncio.run(bridge.broadcast(json.dumps({"type": "fallback", "data": data})))
        return {"status": "fallback-relay", "node": None}

# Example local test
if __name__ == "__main__":
    test_data = {"task": "process", "text": "Hello, Lilith"}
    result = forward_request(test_data)
    print("➡️ Relay Result:", result)
