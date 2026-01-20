import os
import zipfile
import http.server
import socketserver
import json

# Configs
PACK_DIR = './frontend/packs/'
EXTRACT_DIR = './frontend/active_ui/'
MANIFEST_FILE = './config/ship_manifest.json'
DEFAULT_PORTS = {
    "LILLITH": 2626,
    "ORC": 1313,
    "GUARDIAN": 1919
}

def detect_ship():
    ship_name = os.getenv("SHIP_NAME")
    if ship_name:
        print(f"[DAEMON] Ship identified via env: {ship_name}")
        return ship_name.upper()
    try:
        with open(MANIFEST_FILE, 'r') as f:
            manifest = json.load(f)
            ship_name = manifest.get("name", "").upper()
            if ship_name:
                print(f"[DAEMON] Ship identified via manifest: {ship_name}")
                return ship_name
    except Exception as e:
        print(f"[DAEMON] Manifest not found or invalid: {e}")
    return "ORC"

def extract_ui_pack(ship_name):
    zip_path = os.path.join(PACK_DIR, f"{ship_name.lower()}_ui_pack.zip")
    if not os.path.exists(zip_path):
        print(f"[ERROR] UI pack for {ship_name} not found at {zip_path}")
        return False
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"[DAEMON] Extracted {ship_name} UI pack to {EXTRACT_DIR}")
    return True

def run_web_server(port):
    os.chdir(EXTRACT_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"[DAEMON] Serving on port {port}... Press Ctrl+C to stop.")
        httpd.serve_forever()

if __name__ == "__main__":
    ship_name = detect_ship()
    port = DEFAULT_PORTS.get(ship_name, 8080)
    if extract_ui_pack(ship_name):
        run_web_server(port)
