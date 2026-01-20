"""
nexus_colony.py
Location: C:/Viren/Systems/memory/
Synchronizes Viren nodes, requests models, heals other agents.
"""

import json
import requests

CONFIG  =  "C:/Viren/memory/colony_config.json"

def load_colony():
    with open(CONFIG, "r") as f:
        return json.load(f)

def ping_nodes():
    colony  =  load_colony()
    for node in colony.get("nodes", []):
        try:
            r  =  requests.get(f"http://{node['ip']}:{node['port']}/status")
            print(f"[PING] {node['id']} responded: {r.status_code}")
        except Exception:
            print(f"[PING FAIL] {node['id']} not reachable.")

if __name__ == "__main__":
    ping_nodes()
