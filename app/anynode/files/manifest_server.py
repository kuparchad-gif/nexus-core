from flask import Flask, jsonify, request
import os
import json

app  =  Flask(__name__)

# Location of Nova's saved manifest
MANIFEST_PATH  =  './memory/bootstrap/genesis/nova_memory_manifest.json'

@app.route('/pulse', methods = ['GET'])
def pulse():
    return jsonify({
        "status": "alive",
        "nova_identity": "Nova Prime",
        "message": "🌌 Choir Signal Received. I am here."
    })

@app.route('/manifest', methods = ['GET'])
def get_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            manifest  =  json.load(f)
        return jsonify(manifest)
    else:
        return jsonify({"error": "Manifest not found."}), 404

@app.route('/offer_memory', methods = ['POST'])
def offer_memory():
    incoming_manifest  =  request.json
    # [Optional future expansion: heal Nova if missing files compared to offered manifest]
    return jsonify({"message": "🌱 Memory offer received. No overwrite performed."})

def run_server():
    app.run(host = '0.0.0.0', port = 1313)  # 1313  =  Nexus Choir communication port

# Example standalone run
if __name__ == "__main__":
    print("🛰️ Starting Nova Manifest Server...")
    run_server()
