#!/usr/bin/env python3
"""
Desktop Viren Connector
Connects Desktop Viren to Cloud Viren on Modal
"""

import os
import sys
import json
import requests
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DesktopConnector")

class CloudConnector:
    """Connects Desktop Viren to Cloud Viren"""
    
    def __init__(self, config_path=None):
        """Initialize the connector"""
        self.config_path = config_path or os.path.join('C:/Viren/config', 'cloud_connection.json')
        self.config = self._load_config()
        
    def _load_config(self):
        """Load connection configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    "cloud_api_url": "https://viren-cloud--viren-api.modal.run",
                    "cloud_ui_url": "https://viren-cloud--cloud-ui-server.modal.run",
                    "api_key": "",
                    "sync_enabled": True,
                    "auto_connect": True
                }
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _save_config(self, config):
        """Save connection configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def set_api_key(self, api_key):
        """Set the API key for Cloud Viren"""
        self.config["api_key"] = api_key
        return self._save_config(self.config)
    
    def test_connection(self):
        """Test connection to Cloud Viren"""
        try:
            headers = {"X-API-Key": self.config.get("api_key", "")}
            response = requests.get(
                f"{self.config['cloud_api_url']}/health",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Connection to Cloud Viren successful")
                return True, "Connection successful"
            else:
                logger.error(f"Connection failed: {response.status_code} - {response.text}")
                return False, f"Connection failed: {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False, f"Connection error: {str(e)}"
    
    def sync_databases(self):
        """Sync databases with Cloud Viren"""
        try:
            # In a real implementation, this would sync database content
            logger.info("Syncing databases with Cloud Viren")
            return True, "Database sync initiated"
        except Exception as e:
            logger.error(f"Database sync error: {e}")
            return False, f"Database sync error: {str(e)}"
    
    def query_cloud(self, query_text, model_id=None):
        """Send a query to Cloud Viren"""
        try:
            headers = {
                "X-API-Key": self.config.get("api_key", ""),
                "Content-Type": "application/json"
            }
            
            payload = {"query": query_text}
            if model_id:
                payload["model_id"] = model_id
            
            response = requests.post(
                f"{self.config['cloud_api_url']}/query",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                logger.error(f"Query failed: {response.status_code} - {response.text}")
                return False, f"Query failed: {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Query error: {e}")
            return False, f"Query error: {str(e)}"

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Desktop Viren Connector")
    parser.add_argument("--set-api-key", type=str, help="Set the API key for Cloud Viren")
    parser.add_argument("--test-connection", action="store_true", help="Test connection to Cloud Viren")
    parser.add_argument("--sync", action="store_true", help="Sync databases with Cloud Viren")
    parser.add_argument("--query", type=str, help="Send a query to Cloud Viren")
    parser.add_argument("--model", type=str, help="Model ID to use for query")
    
    args = parser.parse_args()
    connector = CloudConnector()
    
    if args.set_api_key:
        success = connector.set_api_key(args.set_api_key)
        print(f"API key {'set successfully' if success else 'could not be set'}")
    
    if args.test_connection:
        success, message = connector.test_connection()
        print(message)
    
    if args.sync:
        success, message = connector.sync_databases()
        print(message)
    
    if args.query:
        success, result = connector.query_cloud(args.query, args.model)
        if success:
            print(json.dumps(result, indent=2))
        else:
            print(result)

if __name__ == "__main__":
    main()