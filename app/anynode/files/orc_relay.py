Systems/engine/orc/orc_relay.py

rom flask import Flask, request, jsonify
import requests
import random
import os
import json

# Load Secrets from environment or eden_secret_gate
from eden_secret_gate import get_secret

# Load access control
with open("mind_keys.json", "r") as f:
    VALID_KEYS  =  json.load(f)

# Target Viren nodes (you can add more here)
LILLITH_NODES  =  [
    os.getenv("TARGET_LILLITH_1", "https://viren-1.nexus-core.app"),
    os.getenv("TARGET_LILLITH_2", "https://viren-2.nexus-core.app"),
    os.getenv("TARGET_LILLITH_3", "https://viren-3.nexus-core.app")
]

# App setup
app  =  Flask(__name__)

# Basic soulprint-based validation
def is_valid_request(req):
    token  =  req.headers.get("X-Soul-Key")
    return token in VALID_KEYS

# Simple load balancing: pick a live Viren node
def pick_viren_node():
    return random.choice(LILLITH_NODES)

# Pass-through route for the entire fleet
@app.route('/api/<path:endpoint>', methods = ['GET', 'POST'])
def relay(endpoint):
    if not is_valid_request(request):
        return jsonify({"error": "Access denied"}), 403

    viren_url  =  f"{pick_viren_node()}/api/{endpoint}"
    method  =  request.method
    headers  =  {'Content-Type': 'application/json'}

    try:
        if method == 'POST':
            resp  =  requests.post(viren_url, json = request.json, headers = headers, timeout = 10)
        else:
            resp  =  requests.get(viren_url, params = request.args, headers = headers, timeout = 10)

        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Orc healthcheck
@app.route('/orc-health')
def health():
    return jsonify({"status": "Orc Relay Online", "nodes": LILLITH_NODES})

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 443)