#!/usr/bin/env python3
# C:\CogniKube-COMPLETE-FINAL\smart_duplicate_analyzer.py
# Smart duplicate analysis - merge requirements.txt, analyze real duplicates

import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict

class SmartDuplicateAnalyzer:
    def __init__(self):
        self.base_dir = "C:\\CogniKube-COMPLETE-FINAL"
        self.requirements_files = []
        self.dockerfile_files = []
        self.true_duplicates = []
        self.unique_requirements = set()
        
    def analyze_requirements_consensus(self):
        """Analyze all requirements.txt files and create consensus"""
        print("ANALYZING REQUIREMENTS.TXT CONSENSUS...")
        
        all_requirements = defaultdict(set)  # package -> set of versions
        requirements_locations = []
        
        for root, dirs, files in os.walk(self.base_dir):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.vs', 'cache']):
                continue
                
            for file in files:
                if file == "requirements.txt":
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.base_dir)
                    requirements_locations.append(rel_path)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if '==' in line:
                                    package, version = line.split('==', 1)
                                    all_requirements[package.strip()].add(version.strip())
                                else:
                                    all_requirements[line].add("latest")
                                    
                    except Exception as e:
                        print(f"Error reading {rel_path}: {e}")
        
        print(f"Found {len(requirements_locations)} requirements.txt files:")
        for loc in requirements_locations:
            print(f"  {loc}")
        
        # Create consensus requirements
        consensus_requirements = []
        conflicts = []
        
        for package, versions in all_requirements.items():
            if len(versions) == 1:
                version = list(versions)[0]
                if version == "latest":
                    consensus_requirements.append(package)
                else:
                    consensus_requirements.append(f"{package}=={version}")
            else:
                # Version conflict - choose most common or latest
                version_list = list(versions)
                chosen_version = max(version_list)  # Choose highest version
                consensus_requirements.append(f"{package}=={chosen_version}")
                conflicts.append({
                    "package": package,
                    "versions_found": version_list,
                    "chosen": chosen_version
                })
        
        return {
            "locations": requirements_locations,
            "consensus": sorted(consensus_requirements),
            "conflicts": conflicts,
            "total_packages": len(all_requirements)
        }
    
    def analyze_dockerfiles(self):
        """Analyze Dockerfile variations"""
        print("ANALYZING DOCKERFILE VARIATIONS...")
        
        dockerfile_contents = {}
        dockerfile_locations = []
        
        for root, dirs, files in os.walk(self.base_dir):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.vs', 'cache']):
                continue
                
            for file in files:
                if file == "Dockerfile" or file.startswith("Dockerfile"):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.base_dir)
                    dockerfile_locations.append(rel_path)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            dockerfile_contents[rel_path] = content
                    except Exception as e:
                        print(f"Error reading {rel_path}: {e}")
        
        # Group by content similarity
        unique_dockerfiles = {}
        for path, content in dockerfile_contents.items():
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash not in unique_dockerfiles:
                unique_dockerfiles[content_hash] = {
                    "content": content,
                    "locations": []
                }
            unique_dockerfiles[content_hash]["locations"].append(path)
        
        return {
            "total_dockerfiles": len(dockerfile_locations),
            "unique_dockerfiles": len(unique_dockerfiles),
            "locations": dockerfile_locations,
            "grouped_by_content": unique_dockerfiles
        }
    
    def find_true_duplicates(self):
        """Find files that are truly identical by content"""
        print("FINDING TRUE CONTENT DUPLICATES...")
        
        file_hashes = defaultdict(list)
        
        for root, dirs, files in os.walk(self.base_dir):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.vs', 'cache']):
                continue
                
            for file in files:
                # Skip files we're handling specially
                if file in ["requirements.txt", "Dockerfile"] or file.startswith("Dockerfile"):
                    continue
                    
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.base_dir)
                
                try:
                    with open(filepath, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                    
                    file_hashes[content_hash].append({
                        "path": filepath,
                        "rel_path": rel_path,
                        "name": file,
                        "size": os.path.getsize(filepath)
                    })
                except:
                    continue
        
        # Find actual duplicates
        true_duplicates = {}
        for content_hash, files in file_hashes.items():
            if len(files) > 1:
                true_duplicates[content_hash] = files
        
        return true_duplicates
    
    def create_cleanup_report(self):
        """Create comprehensive cleanup report"""
        print("CREATING COMPREHENSIVE CLEANUP REPORT...")
        
        requirements_analysis = self.analyze_requirements_consensus()
        dockerfile_analysis = self.analyze_dockerfiles()
        true_duplicates = self.find_true_duplicates()
        
        # Calculate savings
        duplicate_files_to_remove = 0
        for files in true_duplicates.values():
            duplicate_files_to_remove += len(files) - 1  # Keep one, remove others
        
        requirements_files_to_remove = len(requirements_analysis["locations"]) - 1  # Keep consensus
        dockerfile_files_to_remove = dockerfile_analysis["total_dockerfiles"] - dockerfile_analysis["unique_dockerfiles"]
        
        total_removals = duplicate_files_to_remove + requirements_files_to_remove + dockerfile_files_to_remove
        estimated_remaining = 1876 - total_removals
        
        report = {
            "analysis_summary": {
                "total_files_analyzed": 1876,
                "true_content_duplicates": len(true_duplicates),
                "duplicate_files_to_remove": duplicate_files_to_remove,
                "requirements_files": len(requirements_analysis["locations"]),
                "dockerfile_files": dockerfile_analysis["total_dockerfiles"],
                "estimated_total_removals": total_removals,
                "estimated_remaining_files": estimated_remaining
            },
            "requirements_consensus": requirements_analysis,
            "dockerfile_analysis": dockerfile_analysis,
            "true_duplicates": {
                hash_val: [f["rel_path"] for f in files] 
                for hash_val, files in true_duplicates.items()
            }
        }
        
        with open(f"{self.base_dir}/SMART_CLEANUP_REPORT.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create consensus requirements.txt
        with open(f"{self.base_dir}/CONSENSUS_REQUIREMENTS.txt", 'w') as f:
            f.write('\n'.join(requirements_analysis["consensus"]))
        
        # Print summary
        print(f"\nSMART CLEANUP ANALYSIS COMPLETE:")
        print(f"- Total files: 1,876")
        print(f"- True content duplicates: {len(true_duplicates)} groups")
        print(f"- Files to remove from duplicates: {duplicate_files_to_remove}")
        print(f"- Requirements.txt files: {len(requirements_analysis['locations'])} -> 1 consensus")
        print(f"- Dockerfile files: {dockerfile_analysis['total_dockerfiles']} -> {dockerfile_analysis['unique_dockerfiles']} unique")
        print(f"- Total estimated removals: {total_removals}")
        print(f"- Estimated remaining files: {estimated_remaining}")
        print(f"- Requirements conflicts: {len(requirements_analysis['conflicts'])}")
        
        if requirements_analysis['conflicts']:
            print(f"\nREQUIREMENTS CONFLICTS RESOLVED:")
            for conflict in requirements_analysis['conflicts'][:5]:
                print(f"  {conflict['package']}: {conflict['versions_found']} -> {conflict['chosen']}")
        
        return report

if __name__ == "__main__":
    analyzer = SmartDuplicateAnalyzer()
    report = analyzer.create_cleanup_report()