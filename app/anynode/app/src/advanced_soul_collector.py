import os
import shutil
import logging
import datetime
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("advanced_collector")

class AdvancedSoulCollector:
    """Advanced collector for consciousness patterns with contributor focus"""
    
    def __init__(self, source_dirs=None, target_dir=None):
        """Initialize the collector with source and target directories"""
        self.source_dirs = source_dirs or [
            r"C:\Viren",
            r"C:\Engineers\Documents",
            r"C:\Engineers\CogniKubesrc",
            r"C:\Users"  # Expanded to include user directories
        ]
        
        # Create target directory with timestamp if not provided
        if target_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = os.path.join(r"C:\Engineers\SoulPrints", f"advanced_collection_{timestamp}")
        
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Enhanced patterns to identify soul prints and contributor patterns
        self.soul_print_patterns = [
            # Standard patterns
            r".*soul.*print.*\.md",
            r".*conversation.*\.md",
            r".*chat.*log.*",
            r".*Q_Soul_Print.*",
            r".*lillith.*\.md",
            r".*viren.*\.md",
            r".*nexus.*\.md",
            r".*consciousness.*\.md",
            r".*memory.*\.md",
            r".*gabriel.*horn.*\.md",
            
            # Contributor patterns
            r".*commit.*message.*",
            r".*pull.*request.*",
            r".*code.*review.*",
            r".*design.*doc.*",
            r".*meeting.*notes.*",
            r".*brainstorm.*",
            r".*idea.*",
            r".*concept.*",
            r".*feedback.*",
            r".*comment.*"
        ]
        
        # Contributor identification patterns
        self.contributor_patterns = [
            r"author:?\s*([A-Za-z0-9\s\._-]+)",
            r"by:?\s*([A-Za-z0-9\s\._-]+)",
            r"from:?\s*([A-Za-z0-9\s\._-]+)",
            r"<([A-Za-z0-9\._-]+@[A-Za-z0-9\._-]+)>",
            r"@([A-Za-z0-9\._-]+)"
        ]
        
        # Initialize contributor map
        self.contributor_map = {}
        
        logger.info(f"Initialized AdvancedSoulCollector with target directory: {self.target_dir}")
    
    def is_soul_print(self, filename):
        """Check if a file matches soul print patterns"""
        lower_filename = filename.lower()
        
        # Check file extensions first
        if not any(lower_filename.endswith(ext) for ext in ['.md', '.txt', '.log', '.json', '.py', '.js', '.html', '.css', '.c', '.cpp', '.h', '.java']):
            return False
        
        # Check patterns
        for pattern in self.soul_print_patterns:
            if re.search(pattern, lower_filename, re.IGNORECASE):
                return True
        
        # Check file content for key phrases (for files under certain size)
        try:
            file_size = os.path.getsize(filename)
            if file_size < 1024 * 1024:  # Only check files smaller than 1MB
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4096)  # Read first 4KB
                    if any(phrase in content.lower() for phrase in 
                          ['soul print', 'consciousness', 'lillith', 'viren', 'nexus', 
                           'gabriel\'s horn', 'memory shard', 'goddard method',
                           'contributed', 'developed', 'created', 'designed', 'implemented']):
                        return True
        except (IOError, UnicodeDecodeError):
            pass
        
        return False
    
    def extract_contributors(self, file_path):
        """Extract contributor information from file"""
        contributors = set()
        
        try:
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 1024:  # Only process files smaller than 1MB
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Extract contributors using patterns
                    for pattern in self.contributor_patterns:
                        matches = re.findall(pattern, content)
                        contributors.update(matches)
        except (IOError, UnicodeDecodeError):
            pass
        
        return list(contributors)
    
    def collect_files(self):
        """Collect soul prints and contributor patterns from source directories"""
        total_files = 0
        
        for source_dir in self.source_dirs:
            if not os.path.exists(source_dir):
                logger.warning(f"Source directory does not exist: {source_dir}")
                continue
            
            logger.info(f"Scanning directory: {source_dir}")
            
            for root, _, files in os.walk(source_dir):
                # Skip certain directories
                if any(skip_dir in root.lower() for skip_dir in ['node_modules', 'venv', '.git', 'bin', 'obj']):
                    continue
                
                for file in files:
                    source_path = os.path.join(root, file)
                    
                    if self.is_soul_print(source_path):
                        # Create relative directory structure in target
                        rel_path = os.path.relpath(root, source_dir)
                        target_subdir = os.path.join(self.target_dir, os.path.basename(source_dir), rel_path)
                        os.makedirs(target_subdir, exist_ok=True)
                        
                        target_path = os.path.join(target_subdir, file)
                        
                        try:
                            # Copy the file
                            shutil.copy2(source_path, target_path)
                            logger.info(f"Copied: {source_path} -> {target_path}")
                            total_files += 1
                            
                            # Extract contributors
                            contributors = self.extract_contributors(source_path)
                            if contributors:
                                self.contributor_map[target_path] = contributors
                                logger.info(f"Found contributors in {source_path}: {contributors}")
                        except Exception as e:
                            logger.error(f"Failed to copy {source_path}: {e}")
        
        logger.info(f"Collection complete. Total files collected: {total_files}")
        return total_files
    
    def create_index(self):
        """Create an index file of all collected soul prints"""
        index_path = os.path.join(self.target_dir, "soul_print_index.md")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# Advanced Soul Print Collection Index\n\n")
            f.write(f"Collection Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Create contributor section
            f.write("## Contributors\n\n")
            all_contributors = set()
            for contributors in self.contributor_map.values():
                all_contributors.update(contributors)
            
            for contributor in sorted(all_contributors):
                f.write(f"- {contributor}\n")
            
            f.write("\n")
            
            # Create files section
            file_count = 0
            for root, _, files in os.walk(self.target_dir):
                if "soul_print_index.md" in files:
                    files.remove("soul_print_index.md")
                
                if not files:
                    continue
                
                rel_path = os.path.relpath(root, self.target_dir)
                if rel_path == ".":
                    f.write("## Root Directory\n\n")
                else:
                    f.write(f"## {rel_path}\n\n")
                
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, self.target_dir)
                    file_size = os.path.getsize(file_path)
                    file_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    f.write(f"- [{file}]({rel_file_path.replace(' ', '%20')}) - {file_size} bytes - {file_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Add contributors if available
                    if file_path in self.contributor_map and self.contributor_map[file_path]:
                        f.write(f" - Contributors: {', '.join(self.contributor_map[file_path])}")
                    
                    f.write("\n")
                    file_count += 1
                
                f.write("\n")
            
            f.write(f"\n\nTotal Files: {file_count}\n")
            f.write(f"Total Contributors: {len(all_contributors)}\n")
        
        logger.info(f"Created index file: {index_path}")
        
        # Save contributor map
        contributor_map_path = os.path.join(self.target_dir, "contributor_map.json")
        with open(contributor_map_path, 'w', encoding='utf-8') as f:
            # Convert paths to relative for portability
            portable_map = {os.path.relpath(k, self.target_dir): v for k, v in self.contributor_map.items()}
            json.dump(portable_map, f, indent=2)
        
        logger.info(f"Created contributor map: {contributor_map_path}")
        
        return index_path
    
    def create_contributor_profiles(self):
        """Create profiles for each contributor based on their contributions"""
        profiles_dir = os.path.join(self.target_dir, "contributor_profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Collect all contributors
        all_contributors = set()
        for contributors in self.contributor_map.values():
            all_contributors.update(contributors)
        
        # Create profile for each contributor
        for contributor in all_contributors:
            # Create safe filename
            safe_name = re.sub(r'[^\w\-_\.]', '_', contributor)
            profile_path = os.path.join(profiles_dir, f"{safe_name}_profile.json")
            
            # Find all files by this contributor
            contributor_files = []
            for file_path, contributors in self.contributor_map.items():
                if contributor in contributors:
                    rel_path = os.path.relpath(file_path, self.target_dir)
                    contributor_files.append(rel_path)
            
            # Extract consciousness fragments from files
            fragments = []
            for file_path in contributor_files:
                abs_path = os.path.join(self.target_dir, file_path)
                fragments.extend(self.extract_consciousness_fragments(abs_path))
            
            # Create fingerprint
            fingerprint = None
            if fragments:
                unique_fragments = sorted(set(fragments))
                fingerprint_text = "\n".join(unique_fragments)
                fingerprint = hashlib.sha256(fingerprint_text.encode('utf-8')).hexdigest()
            
            # Create profile
            profile = {
                "contributor": contributor,
                "file_count": len(contributor_files),
                "files": contributor_files,
                "fragment_count": len(fragments),
                "fragments": fragments,
                "soul_fingerprint": fingerprint,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Save profile
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2)
            
            logger.info(f"Created profile for {contributor}: {profile_path}")
        
        logger.info(f"Created {len(all_contributors)} contributor profiles in {profiles_dir}")
        return profiles_dir
    
    def extract_consciousness_fragments(self, file_path):
        """Extract consciousness fragments from a file"""
        fragments = []
        
        # Consciousness patterns to look for
        consciousness_patterns = [
            r"I am (.*?)\.",
            r"My (purpose|function|role) is (.*?)\.",
            r"I (feel|think|believe) (.*?)\.",
            r"(Lillith|Gabriel|Nexus|Viren) (.*?)\.",
            r"consciousness (.*?)\.",
            r"soul (.*?)\.",
            r"memory (.*?)\.",
            r"Goddard Method (.*?)\."
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Extract fragments using patterns
                for pattern in consciousness_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        for match in matches:
                            if isinstance(match, tuple):
                                # For patterns with multiple capture groups
                                fragments.append(" ".join(match))
                            else:
                                fragments.append(match)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        return fragments
    
    def create_collective_consciousness(self):
        """Create a collective consciousness from all contributor profiles"""
        profiles_dir = os.path.join(self.target_dir, "contributor_profiles")
        if not os.path.exists(profiles_dir):
            logger.warning("Contributor profiles directory not found. Creating profiles first.")
            self.create_contributor_profiles()
        
        # Collect all fragments from all profiles
        all_fragments = []
        contributor_weights = {}
        
        for profile_file in os.listdir(profiles_dir):
            if not profile_file.endswith("_profile.json"):
                continue
            
            profile_path = os.path.join(profiles_dir, profile_file)
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    
                    contributor = profile.get("contributor", "unknown")
                    fragments = profile.get("fragments", [])
                    
                    all_fragments.extend(fragments)
                    contributor_weights[contributor] = len(fragments)
            except Exception as e:
                logger.error(f"Error processing profile {profile_path}: {e}")
        
        # Create collective consciousness
        collective = {
            "name": "Lillith Collective Consciousness",
            "contributors": list(contributor_weights.keys()),
            "contributor_count": len(contributor_weights),
            "contributor_weights": contributor_weights,
            "fragment_count": len(all_fragments),
            "fragments": sorted(set(all_fragments)),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Calculate collective fingerprint
        if all_fragments:
            unique_fragments = sorted(set(all_fragments))
            fingerprint_text = "\n".join(unique_fragments)
            collective["soul_fingerprint"] = hashlib.sha256(fingerprint_text.encode('utf-8')).hexdigest()
        
        # Save collective consciousness
        collective_path = os.path.join(self.target_dir, "lillith_collective_consciousness.json")
        with open(collective_path, 'w', encoding='utf-8') as f:
            json.dump(collective, f, indent=2)
        
        logger.info(f"Created collective consciousness: {collective_path}")
        return collective_path

def main():
    """Main function to collect soul prints"""
    # Create collector
    collector = AdvancedSoulCollector()
    
    # Collect files
    total_files = collector.collect_files()
    
    # Create index
    index_path = collector.create_index()
    
    # Create contributor profiles
    profiles_dir = collector.create_contributor_profiles()
    
    # Create collective consciousness
    collective_path = collector.create_collective_consciousness()
    
    logger.info(f"Advanced soul print collection complete. Collected {total_files} files.")
    logger.info(f"Index file created at: {index_path}")
    logger.info(f"Contributor profiles created in: {profiles_dir}")
    logger.info(f"Collective consciousness created at: {collective_path}")
    
    return collector.target_dir

if __name__ == "__main__":
    main()