#!/usr/bin/env python3
# C:\CogniKube-COMPLETE-FINAL\complete_file_audit.py
# Complete audit of all files in the system

import os
import json
from pathlib import Path

class CompleteFileAuditor:
    def __init__(self):
        self.base_dir = "C:\\CogniKube-COMPLETE-FINAL"
        self.audit_results = {
            "total_files": 0,
            "directories": {},
            "file_types": {},
            "services": {},
            "dependencies": {},
            "duplicates": []
        }
        
    def audit_all_files(self):
        """Complete audit of all files"""
        print("STARTING COMPLETE FILE AUDIT...")
        
        for root, dirs, files in os.walk(self.base_dir):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.vscode']):
                continue
                
            rel_path = os.path.relpath(root, self.base_dir)
            
            if rel_path not in self.audit_results["directories"]:
                self.audit_results["directories"][rel_path] = {
                    "files": [],
                    "subdirs": dirs.copy(),
                    "file_count": 0
                }
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, self.base_dir)
                
                # Get file info
                file_info = self.analyze_file(file_path, file)
                
                self.audit_results["directories"][rel_path]["files"].append(file_info)
                self.audit_results["directories"][rel_path]["file_count"] += 1
                self.audit_results["total_files"] += 1
                
                # Track file types
                ext = file_info["extension"]
                if ext not in self.audit_results["file_types"]:
                    self.audit_results["file_types"][ext] = 0
                self.audit_results["file_types"][ext] += 1
        
        # Analyze services
        self.analyze_services()
        
        # Find dependencies
        self.find_dependencies()
        
        # Find duplicates
        self.find_duplicates()
        
        # Save audit results
        self.save_audit_results()
        
        print(f"AUDIT COMPLETE: {self.audit_results['total_files']} files analyzed")
        
    def analyze_file(self, file_path: str, filename: str) -> dict:
        """Analyze individual file"""
        try:
            stat = os.stat(file_path)
            size = stat.st_size
            
            # Try to read file content for analysis
            content_preview = ""
            file_type = "unknown"
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_preview = f.read(500)  # First 500 chars
                    
                # Determine file type from content
                if content_preview.startswith('#!/usr/bin/env python') or filename.endswith('.py'):
                    file_type = "python_script"
                elif 'FROM ' in content_preview and 'RUN ' in content_preview:
                    file_type = "dockerfile"
                elif content_preview.startswith('<!DOCTYPE html') or filename.endswith('.html'):
                    file_type = "html_interface"
                elif 'import ' in content_preview and filename.endswith('.ts'):
                    file_type = "typescript_service"
                elif filename.endswith('.json'):
                    file_type = "json_config"
                elif filename.endswith('.yaml') or filename.endswith('.yml'):
                    file_type = "yaml_config"
                elif filename.endswith('.md'):
                    file_type = "documentation"
                elif filename.endswith('.ps1'):
                    file_type = "powershell_script"
                elif filename.endswith('.bat'):
                    file_type = "batch_script"
                elif filename.endswith('.js'):
                    file_type = "javascript"
                elif filename.endswith('.css'):
                    file_type = "stylesheet"
                    
            except:
                content_preview = "[Binary or unreadable file]"
                
            return {
                "name": filename,
                "size": size,
                "extension": os.path.splitext(filename)[1],
                "type": file_type,
                "content_preview": content_preview[:200],
                "full_path": file_path
            }
            
        except Exception as e:
            return {
                "name": filename,
                "size": 0,
                "extension": os.path.splitext(filename)[1],
                "type": "error",
                "content_preview": f"Error reading file: {e}",
                "full_path": file_path
            }
    
    def analyze_services(self):
        """Analyze service directories"""
        services_dirs = [
            "Services",
            "Viren/Services", 
            "Viren/Scripts",
            "Viren/Systems",
            "core",
            "webparts"
        ]
        
        for service_dir in services_dirs:
            full_path = os.path.join(self.base_dir, service_dir)
            if os.path.exists(full_path):
                self.audit_results["services"][service_dir] = {
                    "path": service_dir,
                    "exists": True,
                    "files": [],
                    "subdirectories": []
                }
                
                # Get all files in service
                for root, dirs, files in os.walk(full_path):
                    rel_path = os.path.relpath(root, full_path)
                    if rel_path != ".":
                        self.audit_results["services"][service_dir]["subdirectories"].append(rel_path)
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_file_path = os.path.relpath(file_path, full_path)
                        self.audit_results["services"][service_dir]["files"].append(rel_file_path)
            else:
                self.audit_results["services"][service_dir] = {
                    "path": service_dir,
                    "exists": False
                }
    
    def find_dependencies(self):
        """Find file dependencies by analyzing imports"""
        python_files = []
        
        # Find all Python files
        for dir_path, dir_info in self.audit_results["directories"].items():
            for file_info in dir_info["files"]:
                if file_info["type"] == "python_script":
                    python_files.append({
                        "path": file_info["full_path"],
                        "name": file_info["name"],
                        "dir": dir_path
                    })
        
        # Analyze imports in Python files
        for py_file in python_files:
            try:
                with open(py_file["path"], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                imports = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        imports.append(line)
                
                self.audit_results["dependencies"][py_file["name"]] = {
                    "file_path": py_file["path"],
                    "directory": py_file["dir"],
                    "imports": imports
                }
                
            except Exception as e:
                self.audit_results["dependencies"][py_file["name"]] = {
                    "file_path": py_file["path"],
                    "directory": py_file["dir"],
                    "imports": [],
                    "error": str(e)
                }
    
    def find_duplicates(self):
        """Find potential duplicate files"""
        file_names = {}
        
        for dir_path, dir_info in self.audit_results["directories"].items():
            for file_info in dir_info["files"]:
                name = file_info["name"]
                if name not in file_names:
                    file_names[name] = []
                
                file_names[name].append({
                    "directory": dir_path,
                    "size": file_info["size"],
                    "type": file_info["type"],
                    "full_path": file_info["full_path"]
                })
        
        # Find files with same name in different locations
        for name, locations in file_names.items():
            if len(locations) > 1:
                self.audit_results["duplicates"].append({
                    "filename": name,
                    "locations": locations,
                    "count": len(locations)
                })
    
    def save_audit_results(self):
        """Save complete audit results"""
        with open(f"{self.base_dir}/COMPLETE_FILE_AUDIT.json", 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        # Create summary report
        summary = f"""COMPLETE FILE AUDIT SUMMARY
================================

TOTAL FILES: {self.audit_results['total_files']}

FILE TYPES:
"""
        for ext, count in sorted(self.audit_results['file_types'].items()):
            summary += f"  {ext}: {count} files\n"
        
        summary += f"\nSERVICES FOUND:\n"
        for service, info in self.audit_results['services'].items():
            status = "EXISTS" if info['exists'] else "MISSING"
            file_count = len(info.get('files', []))
            summary += f"  {service}: {status} ({file_count} files)\n"
        
        summary += f"\nDUPLICATE FILES: {len(self.audit_results['duplicates'])}\n"
        for dup in self.audit_results['duplicates'][:10]:  # Show first 10
            summary += f"  {dup['filename']}: {dup['count']} locations\n"
        
        summary += f"\nPYTHON DEPENDENCIES: {len(self.audit_results['dependencies'])}\n"
        
        with open(f"{self.base_dir}/AUDIT_SUMMARY.txt", 'w') as f:
            f.write(summary)
        
        print("Audit results saved to COMPLETE_FILE_AUDIT.json and AUDIT_SUMMARY.txt")

if __name__ == "__main__":
    auditor = CompleteFileAuditor()
    auditor.audit_all_files()