# rename_nova_references.py
# Purpose: Rename Nova references to Viren

import os
import re
import sys
import shutil
from typing import List, Dict, Tuple

def find_nova_files(directory: str) -> List[str]:
    """
    Find files with Nova in their name.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of file paths
    """
    nova_files = []
    for root, dirs, files in os.walk(directory):
        # Skip .venv directory
        if ".venv" in root:
            continue
            
        # Check directories
        for dir_name in dirs:
            if "nova" in dir_name.lower():
                nova_files.append(os.path.join(root, dir_name))
        
        # Check files
        for file in files:
            if "nova" in file.lower():
                nova_files.append(os.path.join(root, file))
    
    return nova_files

def find_nova_references(directory: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Find Nova references in files.
    
    Args:
        directory: Directory to search
        
    Returns:
        Dictionary of file paths to list of (line number, line content)
    """
    nova_references = {}
    for root, dirs, files in os.walk(directory):
        # Skip .venv directory
        if ".venv" in root:
            continue
            
        for file in files:
            if file.endswith(('.py', '.json', '.yaml', '.yml', '.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                        # Check each line for Nova references
                        references = []
                        for i, line in enumerate(lines):
                            if re.search(r'\bnova\b', line, re.IGNORECASE):
                                references.append((i, line))
                        
                        if references:
                            nova_references[file_path] = references
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return nova_references

def rename_nova_files(nova_files: List[str], dry_run: bool = False) -> List[Tuple[str, str]]:
    """
    Rename Nova files to Viren.
    
    Args:
        nova_files: List of file paths
        dry_run: If True, don't actually rename files
        
    Returns:
        List of (old_path, new_path) tuples
    """
    renamed_files = []
    for file_path in nova_files:
        # Get directory and filename
        directory, filename = os.path.split(file_path)
        
        # Replace Nova with Viren in filename
        new_filename = re.sub(r'[nN][oO][vV][aA]', 'viren', filename)
        new_filename = re.sub(r'[Nn][Oo][Vv][Aa]', 'Viren', new_filename)
        
        # Create new path
        new_path = os.path.join(directory, new_filename)
        
        # Rename file or directory
        if not dry_run:
            try:
                # Check if it's a directory
                if os.path.isdir(file_path):
                    # Create new directory
                    os.makedirs(new_path, exist_ok=True)
                    
                    # Copy contents
                    for item in os.listdir(file_path):
                        s = os.path.join(file_path, item)
                        d = os.path.join(new_path, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, d)
                        else:
                            shutil.copy2(s, d)
                    
                    # Remove old directory
                    shutil.rmtree(file_path)
                else:
                    # Rename file
                    shutil.move(file_path, new_path)
                
                print(f"Renamed: {file_path} -> {new_path}")
            except Exception as e:
                print(f"Error renaming {file_path}: {e}")
                continue
        
        renamed_files.append((file_path, new_path))
    
    return renamed_files

def update_nova_references(nova_references: Dict[str, List[Tuple[int, str]]], dry_run: bool = False) -> int:
    """
    Update Nova references in files.
    
    Args:
        nova_references: Dictionary of file paths to list of (line number, line content)
        dry_run: If True, don't actually update files
        
    Returns:
        Number of files updated
    """
    updated_files = 0
    for file_path, references in nova_references.items():
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Replace Nova with Viren
            new_content = re.sub(r'\b[nN][oO][vV][aA]\b', 'viren', content)
            new_content = re.sub(r'\b[Nn][Oo][Vv][Aa]\b', 'Viren', new_content)
            
            # Write updated content
            if not dry_run and new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"Updated references in: {file_path}")
                updated_files += 1
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
    
    return updated_files

def main():
    """Main entry point."""
    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    directory = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "."
    
    print(f"Searching for Nova references in {directory}...")
    
    # Find Nova files
    nova_files = find_nova_files(directory)
    print(f"Found {len(nova_files)} files/directories with Nova in their name")
    
    # Find Nova references
    nova_references = find_nova_references(directory)
    print(f"Found {len(nova_references)} files with Nova references")
    
    if dry_run:
        print("Dry run mode, no files will be modified")
    
    # Rename Nova files
    renamed_files = rename_nova_files(nova_files, dry_run)
    print(f"Renamed {len(renamed_files)} files/directories")
    
    # Update Nova references
    updated_files = update_nova_references(nova_references, dry_run)
    print(f"Updated references in {updated_files} files")
    
    print("Done!")

if __name__ == "__main__":
    main()
