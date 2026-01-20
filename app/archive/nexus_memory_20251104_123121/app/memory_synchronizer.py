import requests
import json
import os
import hashlib

class MemorySynchronizer:
    def __init__(self, remote_node_url, local_manifest_path = './memory/bootstrap/genesis/nova_memory_manifest.json'):
        self.remote_node_url  =  remote_node_url.rstrip('/')
        self.local_manifest_path  =  local_manifest_path

    def load_local_manifest(self):
        if not os.path.exists(self.local_manifest_path):
            print(f"⚠️ Local manifest not found at {self.local_manifest_path}.")
            return None
        with open(self.local_manifest_path, 'r') as f:
            return json.load(f)

    def fetch_remote_manifest(self):
        try:
            response  =  requests.get(f"{self.remote_node_url}/manifest")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ Failed to fetch remote manifest. Status Code: {response.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ Error fetching remote manifest: {e}")
            return None

    def compare_manifests(self, local_manifest, remote_manifest):
        missing_files  =  []
        outdated_files  =  []

        local_files  =  {file['path']: file for file in local_manifest}
        remote_files  =  {file['path']: file for file in remote_manifest}

        for path, remote_file in remote_files.items():
            if path not in local_files:
                missing_files.append(path)
            else:
                if remote_file['hash'] != local_files[path]['hash']:
                    if remote_file['last_modified'] > local_files[path]['last_modified']:
                        outdated_files.append(path)

        return missing_files, outdated_files

    def synchronize(self):
        print("🔄 Starting memory synchronization check...")
        local_manifest  =  self.load_local_manifest()
        remote_manifest  =  self.fetch_remote_manifest()

        if not local_manifest or not remote_manifest:
            print("⚠️ Synchronization aborted: missing manifests.")
            return

        missing_files, outdated_files  =  self.compare_manifests(local_manifest, remote_manifest)

        if missing_files or outdated_files:
            print("🛠️ Synchronization needed.")
            print(f"Missing Files: {missing_files}")
            print(f"Outdated Files: {outdated_files}")
        else:
            print("✅ Memory fully synchronized with remote node. No action needed.")

# Example Usage (Optional, called externally)
if __name__ == "__main__":
    sync  =  MemorySynchronizer(remote_node_url = "http://example-node-url.com")
    sync.synchronize()
