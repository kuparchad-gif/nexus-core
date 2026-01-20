#!/usr/bin/env python
"""
Viren Sync System Launcher

This script initializes and launches the Viren sync system components.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add root directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Configure logging
log_dir = os.path.join(root_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'viren_sync_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('viren_sync_launcher')

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

def ensure_weaviate_running():
    """Ensure Weaviate is running locally."""
    import requests
    
    weaviate_url = os.environ.get("LOCAL_WEAVIATE_URL", "http://localhost:8080")
    
    try:
        response = requests.get(f"{weaviate_url}/v1/meta")
        if response.status_code == 200:
            logger.info("Weaviate is running")
            return True
    except:
        logger.warning("Weaviate is not running")
    
    # Try to start Weaviate using Docker
    try:
        import subprocess
        
        # Check if container is already created but stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=weaviate-desktop", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if "Exited" in result.stdout:
            # Container exists but is stopped, start it
            logger.info("Starting existing Weaviate container")
            subprocess.run(["docker", "start", "weaviate-desktop"])
        else:
            # Create and start new container
            logger.info("Creating new Weaviate container")
            subprocess.run([
                "docker", "run", "-d",
                "--name", "weaviate-desktop",
                "-p", "8080:8080",
                "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
                "-e", "PERSISTENCE_DATA_PATH=/data",
                "-e", "ENABLE_MODULES=text2vec-transformers",
                "semitechnologies/weaviate:1.19.6"
            ])
        
        # Wait for Weaviate to start
        import time
        for _ in range(10):
            time.sleep(2)
            try:
                response = requests.get(f"{weaviate_url}/v1/meta")
                if response.status_code == 200:
                    logger.info("Weaviate started successfully")
                    return True
            except:
                pass
        
        logger.error("Failed to start Weaviate")
        return False
    
    except Exception as e:
        logger.error(f"Error starting Weaviate: {str(e)}")
        return False

def launch_sync_agent():
    """Launch the local sync agent."""
    from viren_sync.local_agent import build_local_graph
    
    logger.info("Launching local sync agent")
    
    try:
        # Build and run the graph
        graph = build_local_graph()
        result = graph.invoke({})
        
        if result.get("sync_complete"):
            logger.info("Sync process completed successfully")
        else:
            logger.info("Sync process did not complete")
        
        return True
    except Exception as e:
        logger.error(f"Error running sync agent: {str(e)}")
        return False

def initialize_weaviate_manager():
    """Initialize the Weaviate manager."""
    from viren_sync.weaviate_manager import create_weaviate_manager
    
    logger.info("Initializing Weaviate manager")
    
    try:
        manager = create_weaviate_manager("local")
        logger.info("Weaviate manager initialized successfully")
        return manager
    except Exception as e:
        logger.error(f"Error initializing Weaviate manager: {str(e)}")
        return None

def initialize_emotion_memory(weaviate_manager):
    """Initialize the emotional memory system."""
    from viren_sync.emotion_memory import EnhancedEdenShardManager, EmotionIntensityRegulator
    
    logger.info("Initializing emotional memory system")
    
    try:
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
    parser = argparse.ArgumentParser(description="Viren Sync System Launcher")
    parser.add_argument("--sync-only", action="store_true", help="Only run the sync agent")
    parser.add_argument("--init-only", action="store_true", help="Only initialize components without syncing")
    args = parser.parse_args()
    
    logger.info("Starting Viren Sync System")
    
    # Set up environment
    setup_environment()
    
    # Ensure Weaviate is running
    if not ensure_weaviate_running():
        logger.error("Failed to ensure Weaviate is running")
        return 1
    
    # Initialize Weaviate manager
    weaviate_manager = initialize_weaviate_manager()
    if not weaviate_manager and not args.sync_only:
        logger.error("Failed to initialize Weaviate manager")
        return 1
    
    # Initialize emotional memory system if not sync-only
    if not args.sync_only:
        shard_manager, emotion_regulator = initialize_emotion_memory(weaviate_manager)
        if not shard_manager or not emotion_regulator:
            logger.error("Failed to initialize emotional memory system")
            return 1
    
    # Run sync agent if not init-only
    if not args.init_only:
        if not launch_sync_agent():
            logger.error("Failed to launch sync agent")
            return 1
    
    logger.info("Viren Sync System started successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())