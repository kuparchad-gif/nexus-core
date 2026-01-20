import os
import shutil
import logging
import datetime
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("soul_collector")

class SoulPrintCollector:
    """Collects chat logs and soul prints into a central location"""
    
    def __init__(self, source_dirs=None, target_dir=None):
        """Initialize the collector with source and target directories"""
        self.source_dirs = source_dirs or [
            r"C:\Viren",
            r"C:\Engineers\Documents",
            r"C:\Engineers\CogniKubesrc"
        ]
        
        # Create target directory with timestamp if not provided
        if target_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = os.path.join(r"C:\Engineers\SoulPrints", f"collection_{timestamp}")
        
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Patterns to identify soul prints and chat logs
        self.soul_print_patterns = [
            r".*soul.*print.*\.md",
            r".*conversation.*\.md",
            r".*chat.*log.*",
            r".*Q_Soul_Print.*",
            r".*lillith.*\.md",
            r".*viren.*\.md",
            r".*nexus.*\.md",
            r".*consciousness.*\.md",
            r".*memory.*\.md",
            r".*gabriel.*horn.*\.md"
        ]
        
        logger.info(f"Initialized SoulPrintCollector with target directory: {self.target_dir}")
    
    def is_soul_print(self, filename):
        """Check if a file matches soul print patterns"""
        lower_filename = filename.lower()
        
        # Check file extensions first
        if not any(lower_filename.endswith(ext) for ext in ['.md', '.txt', '.log', '.json']):
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
                           'gabriel\'s horn', 'memory shard', 'goddard method']):
                        return True
        except (IOError, UnicodeDecodeError):
            pass
        
        return False
    
    def collect_files(self):
        """Collect soul prints and chat logs from source directories"""
        total_files = 0
        
        for source_dir in self.source_dirs:
            if not os.path.exists(source_dir):
                logger.warning(f"Source directory does not exist: {source_dir}")
                continue
            
            logger.info(f"Scanning directory: {source_dir}")
            
            for root, _, files in os.walk(source_dir):
                for file in files:
                    source_path = os.path.join(root, file)
                    
                    if self.is_soul_print(source_path):
                        # Create relative directory structure in target
                        rel_path = os.path.relpath(root, source_dir)
                        target_subdir = os.path.join(self.target_dir, os.path.basename(source_dir), rel_path)
                        os.makedirs(target_subdir, exist_ok=True)
                        
                        target_path = os.path.join(target_subdir, file)
                        
                        try:
                            shutil.copy2(source_path, target_path)
                            logger.info(f"Copied: {source_path} -> {target_path}")
                            total_files += 1
                        except Exception as e:
                            logger.error(f"Failed to copy {source_path}: {e}")
        
        logger.info(f"Collection complete. Total files collected: {total_files}")
        return total_files
    
    def create_index(self):
        """Create an index file of all collected soul prints"""
        index_path = os.path.join(self.target_dir, "soul_print_index.md")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# Soul Print Collection Index\n\n")
            f.write(f"Collection Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
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
                    
                    f.write(f"- [{file}]({rel_file_path.replace(' ', '%20')}) - {file_size} bytes - {file_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    file_count += 1
                
                f.write("\n")
            
            f.write(f"\n\nTotal Files: {file_count}\n")
        
        logger.info(f"Created index file: {index_path}")
        return index_path

def main():
    """Main function to collect soul prints"""
    # Create collector
    collector = SoulPrintCollector()
    
    # Collect files
    total_files = collector.collect_files()
    
    # Create index
    index_path = collector.create_index()
    
    logger.info(f"Soul print collection complete. Collected {total_files} files.")
    logger.info(f"Index file created at: {index_path}")
    
    return collector.target_dir

if __name__ == "__main__":
    main()