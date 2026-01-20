#!/usr/bin/env python
"""
Viren Sync System Initialization

This script initializes the Viren sync system components during boot.
It doesn't run the sync process but ensures all components are properly set up.
"""

import os
import sys
import logging
from datetime import datetime

# Add root directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Configure logging
log_dir = os.path.join(root_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'viren_sync_init_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('viren_sync_init')

def setup_environment():
    """Set up environment variables if not already set."""
    defaults = {
        "LOCAL_WEAVIATE_URL": "http://localhost:8080",
        "CLOUD_WEAVIATE_URL": "https://your-modal-weaviate-url",
        "CLOUD_ENDPOINT": "https://cloud-viren-modal-url/check_cloud_state",
        "LOCAL_TINYLLAMA_PATH": os.path.join(root_dir, "models", "tinyllama-coder-en-v0.1")
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set environment variable {key}={value}")

def initialize_weaviate_manager():
    """Initialize the Weaviate manager."""
    try:
        from viren_sync.weaviate_manager import create_weaviate_manager
        
        logger.info("Initializing Weaviate manager")
        manager = create_weaviate_manager("local")
        logger.info("Weaviate manager initialized successfully")
        return manager
    except Exception as e:
        logger.error(f"Error initializing Weaviate manager: {str(e)}")
        return None

def initialize_emotion_memory(weaviate_manager):
    """Initialize the emotional memory system."""
    try:
        from viren_sync.emotion_memory import EnhancedEdenShardManager, EmotionIntensityRegulator
        
        logger.info("Initializing emotional memory system")
        
        # Set up Redis URL if available
        redis_url = os.environ.get("REDIS_URL")
        
        # Create shard manager
        shard_manager = EnhancedEdenShardManager(redis_url=redis_url, weaviate_manager=weaviate_manager)
        
        # Create emotion regulator
        emotion_regulator = EmotionIntensityRegulator(shard_manager)
        
        logger.info("Emotional memory system initialized successfully")
        return shard_manager, emotion_regulator
    except Exception as e:
        logger.error(f"Error initializing emotional memory system: {str(e)}")
        return None, None

def main():
    """Main entry point."""
    logger.info("Starting Viren Sync System initialization")
    
    # Set up environment
    setup_environment()
    
    # Initialize Weaviate manager
    weaviate_manager = initialize_weaviate_manager()
    if not weaviate_manager:
        logger.warning("Failed to initialize Weaviate manager, continuing with limited functionality")
    
    # Initialize emotional memory system
    shard_manager, emotion_regulator = initialize_emotion_memory(weaviate_manager)
    if not shard_manager or not emotion_regulator:
        logger.warning("Failed to initialize emotional memory system, continuing with limited functionality")
    
    logger.info("Viren Sync System initialization completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())