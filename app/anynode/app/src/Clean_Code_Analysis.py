#!/usr/bin/env python3
# Clean Duplicate Analysis - No interference

import os
from collections import defaultdict

class CleanDuplicateAnalyzer:
    def __init__(self):
        self.base_dir = "C:\\CogniKube-COMPLETE-FINAL"
        
    def analyze_duplicates(self):
        """Find files with same name in different locations"""
        file_groups = defaultdict(list)
        total_files = 0
        
        print("Analyzing files...")
        
        for root, dirs, files in os.walk(self.base_dir):
            # Skip system directories
            if any(skip in root for skip in ['.git', '__pycache__', '.vs', 'cache', 'node_modules']):
                continue
                
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.base_dir)
                
                file_groups[file].append({
                    'full_path': filepath,
                    'rel_path': rel_path,
                    'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
                })
                total_files += 1
        
        # Find actual duplicates
        duplicates = {name: files for name, files in file_groups.items() if len(files) > 1}
        
        return duplicates, total_files
    
    def create_cleanup_report(self, duplicates, total_files):
        """Create cleanup report with correct math"""
        
        # Calculate removal counts
        duplicate_file_count = sum(len(files) for files in duplicates.values())
        files_to_remove = sum(len(files) - 1 for files in duplicates.values())
        files_remaining = total_files - files_to_remove
        
        print(f"\nDUPLICATE ANALYSIS RESULTS:")
        print(f"=" * 40)
        print(f"Total files scanned: {total_files}")
        print(f"Duplicate groups found: {len(duplicates)}")
        print(f"Total duplicate files: {duplicate_file_count}")
        print(f"Files to remove: {files_to_remove}")
        print(f"Files remaining: {files_remaining}")
        print(f"=" * 40)
        
        # Show top duplicates
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"\nTOP 15 DUPLICATE FILES:")
        for i, (name, files) in enumerate(sorted_duplicates[:15]):
            print(f"{i+1:2d}. {name}: {len(files)} copies")
            for j, file_info in enumerate(files[:3]):  # Show first 3 locations
                print(f"    {file_info['rel_path']}")
            if len(files) > 3:
                print(f"    ... and {len(files)-3} more")
        
        # Special analysis for critical files
        critical_files = ['requirements.txt', '__init__.py', 'Dockerfile', 'main.py']
        
        print(f"\nCRITICAL FILE ANALYSIS:")
        for critical in critical_files:
            if critical in duplicates:
                count = len(duplicates[critical])
                print(f"{critical}: {count} copies")
                
        return {
            'total_files': total_files,
            'duplicate_groups': len(duplicates),
            'duplicate_file_count': duplicate_file_count,
            'files_to_remove': files_to_remove,
            'files_remaining': files_remaining,
            'duplicates': duplicates
        }

if __name__ == "__main__":
    analyzer = CleanDuplicateAnalyzer()
    duplicates, total_files = analyzer.analyze_duplicates()
    report = analyzer.create_cleanup_report(duplicates, total_files)
