## File: C:/Viren/Systems/network/viren_mesh.py
```python
"""
viren_mesh.py
Location: C:/Viren/Systems/network/
Handles peer discovery, identity sync, and direct signal communication in the Viren Mesh.
"""

import json
import socket
import threading

class VirenMesh:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.peers = self.config.get("colony_members", [])

    def broadcast_presence(self):
        msg = json.dumps({"node_id": self.config["node_id"], "status": "online"})
        for peer in self.peers:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((peer['ip'], peer['port']))
                s.sendall(msg.encode())
                s.close()
            except:
                print(f"[MESH] Failed to reach {peer['ip']}:{peer['port']}")

    def listen(self):
        def handler():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", self.config["port"]))
                s.listen()
                while True:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        print(f"[MESH] Signal from {addr}: {data.decode()}")
        thread = threading.Thread(target=handler)
        thread.start()
```