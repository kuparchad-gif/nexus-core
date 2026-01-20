#!/usr/bin/env python
"""
Sync Desktop Viren to Cloud Viren on Modal
This script synchronizes the local Viren setup with the cloud deployment on Modal
"""

import os
import sys
import json
import requests
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
        logging.FileHandler(os.path.join('C:/Viren/logs', 'cloud_sync.log'))
    ]
)
logger = logging.getLogger('cloud_sync')

def load_config():
    """Load colony configuration"""
    try:
        config_path = os.path.join('C:/Viren/config', 'colony_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None

def sync_binary_protocol():
    """Sync Binary Protocol data to cloud"""
    try:
        from core.binary_protocol import BinaryProtocol
        
        # Initialize the protocol
        protocol = BinaryProtocol()
        
        # Get cloud endpoints from config
        config = load_config()
        if not config or 'cloud_endpoints' not in config:
            logger.error("No cloud endpoints configured")
            return False
        
        # Sync to each endpoint
        for endpoint in config['cloud_endpoints']:
            url = endpoint['url']
            api_key = endpoint['api_key']
            
            logger.info(f"Syncing to cloud endpoint: {url}")
            
            # Perform sync operation
            success = protocol.sync_to_cloud(url, api_key)
            
            if success:
                logger.info(f"Successfully synced to {url}")
            else:
                logger.error(f"Failed to sync to {url}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to sync Binary Protocol: {e}")
        return False

def sync_weaviate_data():
    """Sync Weaviate data to cloud"""
    try:
        import weaviate
        
        # Get local Weaviate client
        local_client = weaviate.Client("http://localhost:8080")
        
        # Get cloud endpoints from config
        config = load_config()
        if not config or 'cloud_endpoints' not in config:
            logger.error("No cloud endpoints configured")
            return False
        
        # Get vector config
        vector_config_path = os.path.join('C:/Viren/vector', 'weaviate_config.json')
        with open(vector_config_path, 'r') as f:
            vector_config = json.load(f)
        
        # Sync to cloud endpoints
        for endpoint in config['cloud_endpoints']:
            cloud_url = endpoint['url']
            api_key = endpoint['api_key']
            
            # Construct Weaviate URL from API endpoint
            weaviate_url = f"{cloud_url}/weaviate"
            
            logger.info(f"Syncing Weaviate data to: {weaviate_url}")
            
            # Connect to cloud Weaviate
            cloud_client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=api_key)
            )
            
            # Sync each class
            for class_name in vector_config['schema'].values():
                try:
                    # Get objects from local Weaviate
                    result = local_client.query.get(class_name, ["id"]).do()
                    if class_name not in result or 'objects' not in result[class_name]:
                        continue
                    
                    # Get object IDs
                    object_ids = [obj['id'] for obj in result[class_name]['objects']]
                    
                    # Sync each object
                    for obj_id in object_ids:
                        obj = local_client.data_object.get_by_id(obj_id, with_vector=True)
                        if obj:
                            # Check if object exists in cloud
                            try:
                                cloud_client.data_object.get_by_id(obj_id)
                                # Update existing object
                                cloud_client.data_object.update(obj, obj_id, class_name)
                            except:
                                # Create new object
                                cloud_client.data_object.create(obj['properties'], class_name, obj_id, vector=obj['vector'])
                    
                    logger.info(f"Synced {len(object_ids)} objects for class {class_name}")
                except Exception as e:
                    logger.error(f"Error syncing class {class_name}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to sync Weaviate data: {e}")
        return False

def main():
    """Main sync function"""
    logger.info("Starting sync to Cloud Viren")
    
    # Sync Binary Protocol
    if sync_binary_protocol():
        logger.info("Binary Protocol sync completed")
    else:
        logger.error("Binary Protocol sync failed")
    
    # Sync Weaviate data
    if sync_weaviate_data():
        logger.info("Weaviate data sync completed")
    else:
        logger.error("Weaviate data sync failed")
    
    logger.info("Sync complete")

if __name__ == "__main__":
    main()