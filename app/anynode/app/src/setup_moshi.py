#!/usr/bin/env python3
"""
Setup script for Moshi model in Viren
Copies the Moshi model to the Modal volume and configures it for use
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MoshiSetup")

def copy_model_files(source_path: str, modal_volume_name: str = "moshi-model-volume") -> bool:
    """Copy Moshi model files to Modal volume"""
    try:
        # Check if source path exists
        if not os.path.exists(source_path):
            logger.error(f"Source path does not exist: {source_path}")
            return False
        
        # Create temporary directory for model files
        temp_dir = os.path.join(os.path.expanduser("~"), ".modal", "volumes", modal_volume_name)
        os.makedirs(os.path.join(temp_dir, "moshiko-pytorch-bf16"), exist_ok=True)
        
        # Copy model files
        logger.info(f"Copying model files from {source_path} to Modal volume {modal_volume_name}...")
        
        # Copy each file individually
        for file in os.listdir(source_path):
            source_file = os.path.join(source_path, file)
            dest_file = os.path.join(temp_dir, "moshiko-pytorch-bf16", file)
            
            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied {file}")
        
        logger.info("Model files copied successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error copying model files: {e}")
        return False

def install_moshi_library() -> bool:
    """Install Moshi library from GitHub"""
    try:
        logger.info("Installing Moshi library...")
        
        # Clone Moshi repository
        subprocess.run(
            ["git", "clone", "https://github.com/kyutai-labs/moshi.git", "moshi-temp"],
            check=True
        )
        
        # Install Moshi
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "moshi-temp"],
            check=True
        )
        
        logger.info("Moshi library installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Moshi library: {e}")
        return False
    except Exception as e:
        logger.error(f"Error installing Moshi library: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Setup Moshi model for Viren")
    parser.add_argument("--source-path", type=str, default="C:/Engineers/root/models/moshiko-pytorch-bf16",
                        help="Path to Moshi model directory")
    parser.add_argument("--volume-name", type=str, default="moshi-model-volume",
                        help="Name of Modal volume to store model files")
    parser.add_argument("--install-library", action="store_true",
                        help="Install Moshi library from GitHub")
    
    args = parser.parse_args()
    
    # Copy model files
    if not copy_model_files(args.source_path, args.volume_name):
        logger.error("Failed to copy model files")
        sys.exit(1)
    
    # Install Moshi library if requested
    if args.install_library:
        if not install_moshi_library():
            logger.error("Failed to install Moshi library")
            sys.exit(1)
    
    logger.info("Moshi setup completed successfully")
    logger.info("To deploy Moshi to Cloud Viren, run:")
    logger.info("cd C:\\Viren\\cloud")
    logger.info("modal deploy modal_deployment.py")

if __name__ == "__main__":
    main()