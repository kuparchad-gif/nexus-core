#!/usr/bin/env python3
"""
Weaviate Sync Script for Viren
Synchronizes data between local and cloud Weaviate instances
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WeaviateSync")

class WeaviateSync:
    """Synchronizes data between local and cloud Weaviate instances"""
    
    def __init__(
        self,
        cloud_url: str = None,
        local_url: str = None,
        cloud_api_key: str = None,
        local_api_key: str = None,
        sync_interval: int = 3600  # 1 hour default
    ):
        """Initialize the Weaviate sync"""
        self.cloud_url = cloud_url or os.environ.get("VIREN_CLOUD_WEAVIATE_URL", "https://viren-cloud--weaviate-container.modal.run")
        self.local_url = local_url or os.environ.get("VIREN_LOCAL_WEAVIATE_URL", "http://localhost:8080")
        self.cloud_api_key = cloud_api_key or os.environ.get("VIREN_CLOUD_API_KEY", "temporary-key")
        self.local_api_key = local_api_key or os.environ.get("VIREN_LOCAL_API_KEY", "")
        self.sync_interval = sync_interval
        self.last_sync_time = None
        self.sync_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_log.json")
        
        # Load sync log
        self.sync_log = self._load_sync_log()
        
        # Initialize Weaviate clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Weaviate clients"""
        try:
            import weaviate
            
            # Cloud client
            self.cloud_client = weaviate.Client(
                url=self.cloud_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=self.cloud_api_key) if self.cloud_api_key else None
            )
            
            # Local client
            self.local_client = weaviate.Client(
                url=self.local_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=self.local_api_key) if self.local_api_key else None
            )
            
            # Check connections
            if not self.cloud_client.is_ready():
                logger.error("Cloud Weaviate is not ready")
                self.cloud_client = None
            
            if not self.local_client.is_ready():
                logger.error("Local Weaviate is not ready")
                self.local_client = None
            
        except ImportError:
            logger.error("Weaviate client not installed. Install with: pip install weaviate-client")
            self.cloud_client = None
            self.local_client = None
        except Exception as e:
            logger.error(f"Error initializing Weaviate clients: {str(e)}")
            self.cloud_client = None
            self.local_client = None
    
    def _load_sync_log(self) -> Dict:
        """Load sync log from file"""
        try:
            if os.path.exists(self.sync_log_path):
                with open(self.sync_log_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "last_sync": None,
                    "classes": {}
                }
        except Exception as e:
            logger.error(f"Error loading sync log: {str(e)}")
            return {
                "last_sync": None,
                "classes": {}
            }
    
    def _save_sync_log(self):
        """Save sync log to file"""
        try:
            with open(self.sync_log_path, 'w') as f:
                json.dump(self.sync_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sync log: {str(e)}")
    
    def get_schema(self, client) -> Dict:
        """Get schema from Weaviate instance"""
        try:
            return client.schema.get()
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            return {"classes": []}
    
    def sync_schema(self):
        """Synchronize schema between local and cloud instances"""
        if not self.cloud_client or not self.local_client:
            logger.error("Clients not initialized")
            return False
        
        try:
            # Get schemas
            cloud_schema = self.get_schema(self.cloud_client)
            local_schema = self.get_schema(self.local_client)
            
            # Extract class names
            cloud_classes = {cls["class"]: cls for cls in cloud_schema.get("classes", [])}
            local_classes = {cls["class"]: cls for cls in local_schema.get("classes", [])}
            
            # Classes in cloud but not in local
            for class_name, class_obj in cloud_classes.items():
                if class_name not in local_classes:
                    logger.info(f"Creating class {class_name} in local Weaviate")
                    self.local_client.schema.create_class(class_obj)
            
            # Classes in local but not in cloud
            for class_name, class_obj in local_classes.items():
                if class_name not in cloud_classes:
                    logger.info(f"Creating class {class_name} in cloud Weaviate")
                    self.cloud_client.schema.create_class(class_obj)
            
            return True
        except Exception as e:
            logger.error(f"Error syncing schema: {str(e)}")
            return False
    
    def get_objects(self, client, class_name: str, last_update_time: Optional[str] = None) -> List[Dict]:
        """Get objects from a class with optional time filter"""
        try:
            query = client.query.get(class_name, ["id", "_additional {lastUpdateTimeUnix}"])
            
            if last_update_time:
                # Convert to Unix timestamp
                last_update_unix = int(datetime.fromisoformat(last_update_time).timestamp() * 1000)
                query = query.with_where({
                    "path": ["_additional", "lastUpdateTimeUnix"],
                    "operator": "GreaterThan",
                    "valueInt": last_update_unix
                })
            
            result = query.do()
            return result.get("data", {}).get("Get", {}).get(class_name, [])
        except Exception as e:
            logger.error(f"Error getting objects from {class_name}: {str(e)}")
            return []
    
    def get_object_data(self, client, class_name: str, object_id: str) -> Optional[Dict]:
        """Get full object data by ID"""
        try:
            result = client.data_object.get_by_id(object_id, class_name=class_name)
            return result
        except Exception as e:
            logger.error(f"Error getting object data for {class_name}/{object_id}: {str(e)}")
            return None
    
    def sync_class(self, class_name: str):
        """Synchronize objects for a specific class"""
        if not self.cloud_client or not self.local_client:
            logger.error("Clients not initialized")
            return False
        
        try:
            # Get last sync time for this class
            last_sync = self.sync_log.get("classes", {}).get(class_name, {}).get("last_sync")
            
            # Get updated objects from cloud
            logger.info(f"Getting updated objects from cloud for class {class_name}")
            cloud_objects = self.get_objects(self.cloud_client, class_name, last_sync)
            
            # Get updated objects from local
            logger.info(f"Getting updated objects from local for class {class_name}")
            local_objects = self.get_objects(self.local_client, class_name, last_sync)
            
            # Sync cloud to local
            for obj in cloud_objects:
                obj_id = obj.get("id")
                if obj_id:
                    # Get full object data
                    obj_data = self.get_object_data(self.cloud_client, class_name, obj_id)
                    if obj_data:
                        # Check if object exists in local
                        try:
                            local_obj = self.local_client.data_object.get_by_id(obj_id, class_name=class_name)
                            if local_obj:
                                # Update existing object
                                logger.info(f"Updating object {class_name}/{obj_id} in local Weaviate")
                                self.local_client.data_object.update(obj_data, class_name=class_name, uuid=obj_id)
                            else:
                                # Create new object
                                logger.info(f"Creating object {class_name}/{obj_id} in local Weaviate")
                                self.local_client.data_object.create(obj_data, class_name=class_name, uuid=obj_id)
                        except:
                            # Create new object
                            logger.info(f"Creating object {class_name}/{obj_id} in local Weaviate")
                            self.local_client.data_object.create(obj_data, class_name=class_name, uuid=obj_id)
            
            # Sync local to cloud
            for obj in local_objects:
                obj_id = obj.get("id")
                if obj_id:
                    # Get full object data
                    obj_data = self.get_object_data(self.local_client, class_name, obj_id)
                    if obj_data:
                        # Check if object exists in cloud
                        try:
                            cloud_obj = self.cloud_client.data_object.get_by_id(obj_id, class_name=class_name)
                            if cloud_obj:
                                # Update existing object
                                logger.info(f"Updating object {class_name}/{obj_id} in cloud Weaviate")
                                self.cloud_client.data_object.update(obj_data, class_name=class_name, uuid=obj_id)
                            else:
                                # Create new object
                                logger.info(f"Creating object {class_name}/{obj_id} in cloud Weaviate")
                                self.cloud_client.data_object.create(obj_data, class_name=class_name, uuid=obj_id)
                        except:
                            # Create new object
                            logger.info(f"Creating object {class_name}/{obj_id} in cloud Weaviate")
                            self.cloud_client.data_object.create(obj_data, class_name=class_name, uuid=obj_id)
            
            # Update sync log
            now = datetime.now().isoformat()
            if "classes" not in self.sync_log:
                self.sync_log["classes"] = {}
            if class_name not in self.sync_log["classes"]:
                self.sync_log["classes"][class_name] = {}
            self.sync_log["classes"][class_name]["last_sync"] = now
            self.sync_log["last_sync"] = now
            self._save_sync_log()
            
            return True
        except Exception as e:
            logger.error(f"Error syncing class {class_name}: {str(e)}")
            return False
    
    def sync_all(self):
        """Synchronize all classes between local and cloud instances"""
        if not self.cloud_client or not self.local_client:
            logger.error("Clients not initialized")
            return False
        
        try:
            # Sync schema first
            self.sync_schema()
            
            # Get all classes
            cloud_schema = self.get_schema(self.cloud_client)
            cloud_classes = [cls["class"] for cls in cloud_schema.get("classes", [])]
            
            # Sync each class
            for class_name in cloud_classes:
                self.sync_class(class_name)
            
            return True
        except Exception as e:
            logger.error(f"Error syncing all classes: {str(e)}")
            return False
    
    def run_sync_loop(self):
        """Run continuous sync loop"""
        logger.info("Starting sync loop")
        
        while True:
            try:
                # Sync all classes
                logger.info("Running sync")
                self.sync_all()
                
                # Update last sync time
                self.last_sync_time = datetime.now()
                
                # Sleep until next sync
                next_sync = self.last_sync_time + timedelta(seconds=self.sync_interval)
                logger.info(f"Next sync at {next_sync.isoformat()}")
                
                time.sleep(self.sync_interval)
            except KeyboardInterrupt:
                logger.info("Sync loop interrupted")
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Weaviate Sync Tool")
    parser.add_argument("--cloud-url", help="Cloud Weaviate URL")
    parser.add_argument("--local-url", help="Local Weaviate URL")
    parser.add_argument("--cloud-api-key", help="Cloud Weaviate API key")
    parser.add_argument("--local-api-key", help="Local Weaviate API key")
    parser.add_argument("--interval", type=int, default=3600, help="Sync interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run sync once and exit")
    
    args = parser.parse_args()
    
    # Create sync instance
    sync = WeaviateSync(
        cloud_url=args.cloud_url,
        local_url=args.local_url,
        cloud_api_key=args.cloud_api_key,
        local_api_key=args.local_api_key,
        sync_interval=args.interval
    )
    
    if args.once:
        # Run sync once
        sync.sync_all()
    else:
        # Run continuous sync loop
        sync.run_sync_loop()

if __name__ == "__main__":
    main()