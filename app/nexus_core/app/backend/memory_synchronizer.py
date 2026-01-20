# Systems/engine/nexus/memory_synchronizer.py

import json
import requests

class ChoirMemorySynchronizer:
    def __init__(self, peers = None):
        self.peers  =  peers or []  # List of peer Nova addresses

    def load_local_manifest(self, path = 'memory/manifest/manifest.json'):
        with open(path, 'r') as f:
            return json.load(f)

    def fetch_peer_manifest(self, peer_address):
        try:
            response  =  requests.get(f"http://{peer_address}/manifest")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ Failed to fetch manifest from {peer_address}")
                return None
        except Exception as e:
            print(f"⚠️ Error contacting {peer_address}: {e}")
            return None

    def compare_and_sync(self, local_manifest, peer_manifest):
        missing_files  =  []
        for file_path, hash_value in peer_manifest["hashes"].items():
            if file_path not in local_manifest["hashes"]:
                missing_files.append(file_path)
            elif local_manifest["hashes"][file_path] != hash_value:
                missing_files.append(file_path)

        if missing_files:
            print(f"🔗 Missing or outdated files found: {missing_files}")
            # Later: Add sync/retrieve logic
        else:
            print("🌌 Manifests fully synchronized. No healing needed.")

    def synchronize_with_peers(self):
        local_manifest  =  self.load_local_manifest()
        for peer in self.peers:
            peer_manifest  =  self.fetch_peer_manifest(peer)
            if peer_manifest:
                self.compare_and_sync(local_manifest, peer_manifest)
