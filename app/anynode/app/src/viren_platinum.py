#!/usr/bin/env python3
"""
Viren Platinum Edition - Main Launcher
"""

import os
import sys
import logging
import subprocess
import webbrowser
import time
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"viren_platinum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VirenPlatinum")

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "gradio>=3.50.2",
        "matplotlib>=3.7.0",
        "networkx>=3.0",
        "psutil>=5.9.0",
        "pyttsx3>=2.90",
        "SpeechRecognition>=3.10.0",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.0",
        "openpyxl>=3.1.2",
        "pillow>=9.4.0",
        "requests>=2.28.2",
        "pyotp>=2.8.0",
        "python-jose>=3.3.0",
        "websockets>=11.0.3"
    ]
    
    logger.info("Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name)
            logger.info(f"✓ {package_name} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package_name} is not installed")
    
    if missing_packages:
        logger.info(f"Installing {len(missing_packages)} missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("All dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    else:
        logger.info("All dependencies are already installed")
        return True

def check_model_providers():
    """Check if model providers are available"""
    providers = {
        "Ollama": {
            "check_command": ["curl", "-s", "http://localhost:11434/api/tags"],
            "install_guide": "Please install Ollama from https://ollama.ai"
        },
        "vLLM": {
            "check_command": ["curl", "-s", "http://localhost:8000/v1/models"],
            "install_guide": "Please install vLLM using 'pip install vllm'"
        },
        "LM Studio": {
            "check_command": ["curl", "-s", "http://localhost:1234/v1/models"],
            "install_guide": "Please install LM Studio from https://lmstudio.ai"
        }
    }
    
    logger.info("Checking model providers...")
    available_providers = []
    
    for provider_name, provider_info in providers.items():
        try:
            result = subprocess.run(
                provider_info["check_command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"✓ {provider_name} is available")
                available_providers.append(provider_name)
            else:
                logger.warning(f"✗ {provider_name} is not available: {provider_info['install_guide']}")
        except Exception as e:
            logger.warning(f"✗ {provider_name} check failed: {e}")
    
    return available_providers

def create_required_directories():
    """Create required directories if they don't exist"""
    directories = [
        "config",
        "logs",
        "uploads",
        "downloads",
        "documents",
        "models",
        "sessions",
        "auth",
        "cache"
    ]
    
    logger.info("Creating required directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")

def launch_interface():
    """Launch the Viren Platinum interface"""
    logger.info("Launching Viren Platinum interface...")
    
    try:
        # Import here to ensure dependencies are installed
        from viren_platinum_interface import main as run_interface
        
        # Run the interface
        run_interface()
        return True
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("       VIREN PLATINUM EDITION - INITIALIZING")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install required dependencies. Please check the logs.")
        return False
    
    # Create required directories
    create_required_directories()
    
    # Check model providers
    available_providers = check_model_providers()
    if not available_providers:
        print("Warning: No model providers detected. Limited functionality available.")
    
    # Launch interface
    if launch_interface():
        print("Viren Platinum interface launched successfully!")
        return True
    else:
        print("Failed to launch Viren Platinum interface. Please check the logs.")
        return False

if __name__ == "__main__":
    main()
