#!/usr/bin/env python
"""
Weaviate Setup Script for Viren
This script initializes the Weaviate vector database with the necessary schema
"""

import os
import sys
import json
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
        logging.FileHandler(os.path.join('C:/Viren/logs', 'weaviate_setup.log'))
    ]
)
logger = logging.getLogger('weaviate_setup')

def load_config():
    """Load Weaviate configuration"""
    try:
        config_path = os.path.join('C:/Viren/vector', 'weaviate_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None

def setup_weaviate_client():
    """Set up and return a Weaviate client"""
    try:
        import weaviate
        from weaviate.auth import AuthApiKey
        
        config = load_config()
        if not config:
            return None
        
        weaviate_url = config['weaviate']['url']
        api_key = config['weaviate']['api_key']
        
        auth = None
        if api_key:
            auth = AuthApiKey(api_key=api_key)
        
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=auth,
            timeout_config=(
                config['weaviate']['timeout_config']['query'],
                config['weaviate']['timeout_config']['insert']
            )
        )
        
        if client.is_ready():
            logger.info(f"Connected to Weaviate at {weaviate_url}")
            return client
        else:
            logger.error(f"Weaviate at {weaviate_url} is not ready")
            return None
    except ImportError:
        logger.error("Weaviate client not installed. Run: pip install weaviate-client")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        return None

def create_schema(client):
    """Create the Weaviate schema"""
    try:
        from vector.weaviate_schema import get_schema
        
        schema = get_schema()
        
        # Check if schema already exists
        existing_classes = {cls['class'] for cls in client.schema.get()['classes']} if client.schema.get()['classes'] else set()
        
        for class_obj in schema['classes']:
            class_name = class_obj['class']
            if class_name in existing_classes:
                logger.info(f"Class {class_name} already exists, skipping")
                continue
            
            client.schema.create_class(class_obj)
            logger.info(f"Created class {class_name}")
        
        logger.info("Schema creation completed")
        return True
    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Weaviate setup")
    
    # Set up Weaviate client
    client = setup_weaviate_client()
    if not client:
        logger.error("Failed to set up Weaviate client")
        sys.exit(1)
    
    # Create schema
    if create_schema(client):
        logger.info("Weaviate setup completed successfully")
    else:
        logger.error("Weaviate setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()