# Systems/engine/launch_viren.py
# Purpose: Launch script for Viren

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/launch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("launch_viren")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch Viren")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory loading")
    parser.add_argument("--no-advanced", action="store_true", help="Disable advanced features")
    return parser.parse_args()

def load_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # Import here to avoid circular imports
    from Systems.Config.viren_runtime import viren_runtime
    
    if config_path:
        # Override the default configuration path
        viren_runtime.config_path = config_path
        viren_runtime.config = viren_runtime._load_config()
    
    return viren_runtime.get_config()

def initialize_memory(config: Dict[str, Any], disable_memory: bool = False) -> bool:
    """
    Initialize the memory system.
    
    Args:
        config: Configuration dictionary
        disable_memory: Whether to disable memory loading
        
    Returns:
        True if successful, False otherwise
    """
    if disable_memory:
        logger.info("Memory loading disabled")
        return True
    
    if not config.get("services", {}).get("memory", {}).get("enabled", True):
        logger.info("Memory service disabled in configuration")
        return True
    
    try:
        # Import the memory loader
        from memory.bootstrap.viren_memory_loader import memory_loader
        
        # Load all corpora
        success = memory_loader.load_all_corpora()
        
        # Process incoming training sets
        if success:
            success = memory_loader.process_incoming_training_sets()
        
        if success:
            logger.info("Memory initialization successful")
        else:
            logger.warning("Memory initialization completed with warnings")
        
        return success
    except Exception as e:
        logger.error(f"Error initializing memory: {e}")
        return False

def initialize_advanced_features(config: Dict[str, Any], disable_advanced: bool = False) -> bool:
    """
    Initialize advanced features.
    
    Args:
        config: Configuration dictionary
        disable_advanced: Whether to disable advanced features
        
    Returns:
        True if successful, False otherwise
    """
    if disable_advanced:
        logger.info("Advanced features disabled")
        return True
    
    if not config.get("advanced", {}).get("enabled", True):
        logger.info("Advanced features disabled in configuration")
        return True
    
    try:
        # Import the Viren brain
        from Services.viren_brain import viren_brain
        
        # Initialize the brain
        success = viren_brain.initialize()
        
        if success:
            logger.info("Advanced features initialization successful")
        else:
            logger.warning("Advanced features initialization completed with warnings")
        
        return success
    except Exception as e:
        logger.error(f"Error initializing advanced features: {e}")
        return False

def start_services(config: Dict[str, Any]) -> bool:
    """
    Start all services.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import the bootstrap module
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from bootstrap_viren import start_services as bootstrap_start_services
        
        # Start services
        success = bootstrap_start_services()
        
        if success:
            logger.info("Services started successfully")
        else:
            logger.warning("Services started with warnings")
        
        return success
    except Exception as e:
        logger.error(f"Error starting services: {e}")
        return False

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info("Starting Viren launch sequence")
    
    # Load configuration
    config = load_configuration(args.config)
    logger.info(f"Loaded configuration version {config.get('version', 'unknown')}")
    
    # Initialize memory
    if not initialize_memory(config, args.no_memory):
        logger.error("Memory initialization failed")
        return 1
    
    # Initialize advanced features
    if not initialize_advanced_features(config, args.no_advanced):
        logger.warning("Advanced features initialization failed, continuing with basic features")
    
    # Start services
    if not start_services(config):
        logger.error("Failed to start services")
        return 1
    
    logger.info("Viren launch sequence completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())