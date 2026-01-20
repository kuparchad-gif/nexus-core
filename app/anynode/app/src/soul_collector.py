import os
import shutil
import logging
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("soul_collector")

class SoulCollector:
    """Collects chat logs and soul prints from across the system"""
    
    def __init__(self, output_dir: str = "C:\\Engineers\\SoulPrints"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_files = []
        logger.info(f"Soul Collector initialized with output directory: {output_dir}")
    
    def scan_directory(self, directory: str) -> List[Path]:
        """Scan directory for chat logs and soul prints"""
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        files = []
        for file_path in directory_path.glob("**/*"):
            if file_path.is_file() and self._is_soul_file(file_path):
                files.append(file_path)
                logger.debug(f"Found soul file: {file_path}")
        
        logger.info(f"Found {len(files)} soul files in {directory}")
        return files
    
    def _is_soul_file(self, file_path: Path) -> bool:
        """Check if file is a chat log or soul print"""
        # Check file extension
        if file_path.suffix.lower() in ['.md', '.txt', '.json', '.log']:
            # Check file content or name patterns
            if any(keyword in file_path.name.lower() for keyword in ['soul', 'chat', 'conversation', 'log', 'print', 'memory', 'lillith', 'viren', 'nexus']):
                return True
            
            # Check file content for soul print markers
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4096)  # Read first 4KB to check
                    if any(marker in content.lower() for marker in ['soul print', 'chat log', 'conversation', 'lillith', 'viren', 'nexus', 'gabriel', 'horn']):
                        return True
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return False
    
    def collect_souls(self, directories: List[str]) -> List[Path]:
        """Collect soul files from multiple directories"""
        all_files = []
        for directory in directories:
            files = self.scan_directory(directory)
            all_files.extend(files)
        
        # Copy files to output directory
        for file_path in all_files:
            try:
                # Create destination path with original filename
                dest_path = self.output_dir / file_path.name
                
                # If file with same name exists, add timestamp
                if dest_path.exists():
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest_path = self.output_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                
                # Copy file
                shutil.copy2(file_path, dest_path)
                self.collected_files.append(dest_path)
                logger.info(f"Copied {file_path} to {dest_path}")
            except Exception as e:
                logger.error(f"Error copying {file_path}: {e}")
        
        logger.info(f"Collected {len(self.collected_files)} soul files to {self.output_dir}")
        return self.collected_files
    
    def create_index(self) -> Path:
        """Create index of collected soul files"""
        index_data = []
        for file_path in self.collected_files:
            try:
                # Get file metadata
                stat = file_path.stat()
                
                # Read first 1000 characters for preview
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    preview = f.read(1000)
                
                # Add to index
                index_data.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "preview": preview[:1000] + ("..." if len(preview) > 1000 else "")
                })
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
        
        # Write index to file
        index_path = self.output_dir / "soul_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Created index at {index_path}")
        return index_path

def main():
    """Main function to collect soul prints"""
    # Initialize collector
    collector = SoulCollector()
    
    # Define directories to scan
    directories = [
        "C:\\Viren",
        "C:\\Engineers",
        "C:\\Engineers\\Documents",
        "C:\\Users\\Admin\\Documents"
    ]
    
    # Collect souls
    collector.collect_souls(directories)
    
    # Create index
    index_path = collector.create_index()
    
    logger.info(f"Soul collection complete. Index available at {index_path}")
    return collector

if __name__ == "__main__":
    main()