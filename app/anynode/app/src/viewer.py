
# Read-only file viewer interface for Nexus console
import os

class ReadOnlyViewer:
    def __init__(self, base_path='memory'):
        self.base_path = base_path

    def list_files(self):
        result = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                path = os.path.relpath(os.path.join(root, file), self.base_path)
                result.append(path)
        return result

    def read_file(self, relative_path):
        full_path = os.path.join(self.base_path, relative_path)
        if not os.path.exists(full_path):
            return "File not found."
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
