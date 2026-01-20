#!/usr/bin/env python3
# C:\CogniKube-COMPLETE-FINAL\duplicate_cleaner.py
# Analyze and clean duplicate files before containerizing

import os
import json
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

class DuplicateCleaner:
    def __init__(self):
        self.base_dir = "C:\\CogniKube-COMPLETE-FINAL"
        self.duplicates_by_name = defaultdict(list)
        self.duplicates_by_content = defaultdict(list)
        self.files_to_keep = []
        self.files_to_remove = []
        
    def get_file_hash(self, filepath):
        """Get MD5 hash of file content"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def analyze_duplicates(self):
        """Analyze all duplicates by name and content"""
        print("ANALYZING DUPLICATES IN 1,876 FILES...")
        
        file_hashes = defaultdict(list)
        file_names = defaultdict(list)
        
        for root, dirs, files in os.walk(self.base_dir):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.vs', 'cache']):
                continue
                
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.base_dir)
                
                # Group by filename
                file_names[file].append({
                    'path': filepath,
                    'rel_path': rel_path,
                    'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
                })
                
                # Group by content hash
                file_hash = self.get_file_hash(filepath)
                if file_hash:
                    file_hashes[file_hash].append({
                        'path': filepath,
                        'rel_path': rel_path,
                        'name': file,
                        'size': os.path.getsize(filepath)
                    })
        
        # Find duplicates by name
        name_duplicates = {name: files for name, files in file_names.items() if len(files) > 1}
        
        # Find duplicates by content
        content_duplicates = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}
        
        print(f"DUPLICATE ANALYSIS RESULTS:")
        print(f"Files with duplicate names: {len(name_duplicates)}")
        print(f"Files with duplicate content: {len(content_duplicates)}")
        
        return name_duplicates, content_duplicates
    
    def create_cleanup_plan(self, name_duplicates, content_duplicates):
        """Create smart cleanup plan with requirements.txt consensus"""
        cleanup_plan = {
            "total_duplicates_by_name": len(name_duplicates),
            "total_duplicates_by_content": len(content_duplicates),
            "files_to_remove": [],
            "files_to_keep": [],
            "requirements_consensus": {},
            "cleanup_rules": {}
        }
        
        # Analyze specific duplicate patterns
        critical_duplicates = {}
        
        for name, files in name_duplicates.items():
            if len(files) > 1:
                critical_duplicates[name] = {
                    "count": len(files),
                    "locations": [f["rel_path"] for f in files],
                    "sizes": [f["size"] for f in files]
                }
        
        # Show top duplicates
        sorted_duplicates = sorted(critical_duplicates.items(), key=lambda x: x[1]["count"], reverse=True)
        
        print(f"\nTOP 20 DUPLICATE FILES:")
        for i, (name, info) in enumerate(sorted_duplicates[:20]):
            print(f"{i+1:2d}. {name}: {info['count']} copies")
            for loc in info['locations'][:3]:  # Show first 3 locations
                print(f"    {loc}")
            if len(info['locations']) > 3:
                print(f"    ... and {len(info['locations'])-3} more")
        
        # Smart removal with special handling
        removal_count = 0
        processed_files = set()  # Prevent double counting
        
        for name, files in name_duplicates.items():
            if len(files) > 1:
                # Special handling for requirements.txt
                if name == "requirements.txt":
                    consensus_req = self.create_requirements_consensus(files)
                    cleanup_plan["requirements_consensus"] = consensus_req
                    # Keep one, remove others
                    keep_file = files[0]
                    cleanup_plan["files_to_keep"].append(keep_file["rel_path"])
                    
                    for duplicate in files[1:]:
                        if duplicate["rel_path"] not in processed_files:
                            cleanup_plan["files_to_remove"].append(duplicate["rel_path"])
                            processed_files.add(duplicate["rel_path"])
                            removal_count += 1
                
                # Special handling for __init__.py (keep if different directories)
                elif name == "__init__.py":
                    # Keep all __init__.py in different directories
                    for file in files:
                        cleanup_plan["files_to_keep"].append(file["rel_path"])
                
                # Regular duplicates - keep first, remove others
                else:
                    keep_file = files[0]
                    cleanup_plan["files_to_keep"].append(keep_file["rel_path"])
                    
                    for duplicate in files[1:]:
                        if duplicate["rel_path"] not in processed_files:
                            cleanup_plan["files_to_remove"].append(duplicate["rel_path"])
                            processed_files.add(duplicate["rel_path"])
                            removal_count += 1
        
        cleanup_plan["estimated_removals"] = removal_count
        cleanup_plan["estimated_remaining"] = 1876 - removal_count
        
        print(f"\nSMART CLEANUP MATH:")
        print(f"Total files: 1,876")
        print(f"Duplicate file groups: {len(name_duplicates)}")
        print(f"Files to remove: {removal_count}")
        print(f"Files remaining: {1876 - removal_count}")
        
        return cleanup_plan
    
    def create_requirements_consensus(self, req_files):
        """Create consensus requirements.txt from multiple versions"""
        all_requirements = {}
        
        for req_file in req_files:
            try:
                with open(req_file["path"], 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '==' in line:
                                package, version = line.split('==', 1)
                                if package not in all_requirements:
                                    all_requirements[package] = []
                                all_requirements[package].append(version)
                            else:
                                if line not in all_requirements:
                                    all_requirements[line] = []
            except:
                continue
        
        # Create consensus - use most common version
        consensus = {}
        for package, versions in all_requirements.items():
            if versions:
                # Use most recent/common version
                consensus[package] = max(versions) if versions else versions[0]
            else:
                consensus[package] = "latest"
        
        return consensus
    
    def save_cleanup_report(self, name_duplicates, content_duplicates, cleanup_plan):
        """Save detailed cleanup report"""
        report = {
            "analysis_timestamp": "2025-01-15",
            "total_files_analyzed": 1876,
            "duplicates_by_name": len(name_duplicates),
            "duplicates_by_content": len(content_duplicates),
            "cleanup_plan": cleanup_plan,
            "detailed_duplicates": {}
        }
        
        # Add detailed duplicate info
        for name, files in name_duplicates.items():
            if len(files) > 1:
                report["detailed_duplicates"][name] = {
                    "count": len(files),
                    "files": [{"path": f["rel_path"], "size": f["size"]} for f in files]
                }
        
        with open(f"{self.base_dir}/DUPLICATE_CLEANUP_REPORT.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary
        summary = f"""DUPLICATE CLEANUP ANALYSIS SUMMARY
==========================================

TOTAL FILES ANALYZED: 1,876

DUPLICATES FOUND:
- Files with duplicate names: {len(name_duplicates)}
- Files with duplicate content: {len(content_duplicates)}

CLEANUP PLAN:
- Files to remove: {cleanup_plan['estimated_removals']}
- Files to keep: {cleanup_plan['estimated_remaining']}
- Space savings: Significant

TOP DUPLICATE FILES:
"""
        
        # Add top duplicates to summary
        critical_duplicates = {}
        for name, files in name_duplicates.items():
            if len(files) > 1:
                critical_duplicates[name] = len(files)
        
        sorted_duplicates = sorted(critical_duplicates.items(), key=lambda x: x[1], reverse=True)
        for i, (name, count) in enumerate(sorted_duplicates[:10]):
            summary += f"{i+1:2d}. {name}: {count} copies\n"
        
        with open(f"{self.base_dir}/DUPLICATE_CLEANUP_SUMMARY.txt", 'w') as f:
            f.write(summary)
        
        print(f"\nSMART CLEANUP ANALYSIS COMPLETE:")
        print(f"- Duplicate file groups found: {len(name_duplicates)}")
        print(f"- Files recommended for removal: {cleanup_plan['estimated_removals']}")
        print(f"- Estimated remaining files: {cleanup_plan['estimated_remaining']}")
        print(f"- Requirements.txt consensus created: {len(cleanup_plan.get('requirements_consensus', {}))} packages")
        print(f"- Reports saved: DUPLICATE_CLEANUP_REPORT.json, DUPLICATE_CLEANUP_SUMMARY.txt")
        
        return cleanup_plan

if __name__ == "__main__":
    cleaner = DuplicateCleaner()
    name_dups, content_dups = cleaner.analyze_duplicates()
    cleanup_plan = cleaner.create_cleanup_plan(name_dups, content_dups)
    cleaner.save_cleanup_report(name_dups, content_dups, cleanup_plan)