#!/usr/bin/env python
"""
Binary Protocol Installation Script for Viren Desktop
This script initializes the Binary Protocol system for Viren Desktop
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('C:/Viren/logs', 'binary_protocol_install.log'))
    ]
)
logger = logging.getLogger('binary_protocol_installer')

def create_directory_structure():
    """Create necessary directory structure for Binary Protocol"""
    logger.info("Creating directory structure for Binary Protocol")
    
    directories = [
        "C:/Viren/memory/shards/primary",
        "C:/Viren/memory/shards/secondary",
        "C:/Viren/memory/shards/tertiary",
        "C:/Viren/config/keys",
        "C:/Viren/logs/binary_protocol"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def initialize_binary_protocol():
    """Initialize the Binary Protocol system"""
    logger.info("Initializing Binary Protocol")
    
    try:
        # Import necessary modules
        from core.binary_protocol import BinaryProtocol
        
        # Initialize the protocol
        protocol = BinaryProtocol()
        protocol.initialize()
        
        logger.info("Binary Protocol initialized successfully")
        return True
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Binary Protocol: {e}")
        return False

def sync_with_cloud():
    """Sync configuration with Cloud Viren if available"""
    logger.info("Checking for Cloud Viren configuration")
    
    cloud_zip = Path("C:/Viren/Cloudviren.zip")
    if cloud_zip.exists():
        try:
            import zipfile
            
            # Extract only configuration files
            with zipfile.ZipFile(cloud_zip, 'r') as zip_ref:
                config_files = [f for f in zip_ref.namelist() if f.startswith('config/')]
                for file in config_files:
                    zip_ref.extract(file, "C:/Viren/temp_cloud")
            
            # Copy extracted config files
            cloud_config_dir = Path("C:/Viren/temp_cloud/config")
            if cloud_config_dir.exists():
                for file in cloud_config_dir.glob('*'):
                    if file.name.endswith('.json'):
                        shutil.copy(file, Path("C:/Viren/config") / file.name)
                        logger.info(f"Synced config file: {file.name}")
            
            # Clean up
            shutil.rmtree("C:/Viren/temp_cloud", ignore_errors=True)
            logger.info("Cloud configuration sync completed")
        except Exception as e:
            logger.error(f"Failed to sync with Cloud Viren: {e}")
    else:
        logger.info("Cloud Viren configuration not found, using local configuration")

def main():
    """Main installation function"""
    logger.info("Starting Binary Protocol installation")
    
    # Create directory structure
    create_directory_structure()
    
    # Sync with cloud if available
    sync_with_cloud()
    
    # Initialize Binary Protocol
    if initialize_binary_protocol():
        logger.info("Binary Protocol installation completed successfully")
    else:
        logger.error("Binary Protocol installation failed")
        sys.exit(1)
    
    logger.info("Installation complete")

if __name__ == "__main__":
    main()