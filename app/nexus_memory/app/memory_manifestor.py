import os
import json
import hashlib
import time

class MemoryManifestor:
    def __init__(self, root_folder = '.', output_folder = './memory/bootstrap/genesis/'):
        self.root_folder  =  root_folder
        self.output_folder  =  output_folder
        self.manifest  =  []

    def generate_manifest(self):
        for dirpath, _, filenames in os.walk(self.root_folder):
            for file_name in filenames:
                if file_name.endswith(('.py', '.yaml', '.json', '.md', '.txt', '.csv')):
                    full_path  =  os.path.join(dirpath, file_name)
                    try:
                        file_stat  =  os.stat(full_path)
                        file_info  =  {
                            "path": full_path.replace(self.root_folder, ''),
                            "size": file_stat.st_size,
                            "last_modified": time.ctime(file_stat.st_mtime),
                            "hash": self.calculate_file_hash(full_path)
                        }
                        self.manifest.append(file_info)
                    except Exception as e:
                        print(f"⚠️ Error reading {full_path}: {e}")

    def calculate_file_hash(self, filepath):
        hasher  =  hashlib.sha256()
        with open(filepath, 'rb') as f:
            buf  =  f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def save_manifest(self):
        os.makedirs(self.output_folder, exist_ok = True)
        output_file  =  os.path.join(self.output_folder, 'nova_memory_manifest.json')
        with open(output_file, 'w') as f:
            json.dump(self.manifest, f, indent = 4)
        print(f"🌟 Memory Manifest Saved: {output_file}")

    def run(self):
        print("🧠 Generating Nova Memory Manifest...")
        self.generate_manifest()
        self.save_manifest()

# Example Usage (for Bootstrap Integration)
if __name__ == "__main__":
    manifestor  =  MemoryManifestor(root_folder = '.')
    manifestor.run()
