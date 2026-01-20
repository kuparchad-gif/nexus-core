# Check for remaining Nova references
import os
import sys
import re

def find_nova_refs(directory):
    nova_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.json', '.yaml', '.yml', '.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'nova' in content.lower():
                            nova_files.append(file_path)
                except Exception:
                    pass
    return nova_files

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    nova_files = find_nova_refs(directory)
    print(f'Found {len(nova_files)} files with Nova references:')
    for file in nova_files:
        print(f'  {file}')