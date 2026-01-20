#!/usr/bin/env python3
"""
Sync Status Monitor for Cloud Viren
Monitors and reports on the synchronization status between Desktop and Cloud Viren
"""

import os
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SyncMonitor")

class SyncStatusMonitor:
    """Monitors synchronization status between Desktop and Cloud Viren"""
    
    def __init__(self, status_path=None):
        """Initialize the sync status monitor"""
        self.status_path = status_path or os.path.join('C:/Viren/data', 'sync_status.json')
        self.status = self._load_status()
    
    def _load_status(self):
        """Load the sync status"""
        try:
            if os.path.exists(self.status_path):
                with open(self.status_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default status
                default_status = {
                    "last_sync": 0,
                    "sync_in_progress": False,
                    "systems": {},
                    "databases": {},
                    "models": {}
                }
                self._save_status(default_status)
                return default_status
        except Exception as e:
            logger.error(f"Error loading sync status: {e}")
            return {
                "last_sync": 0,
                "sync_in_progress": False,
                "systems": {},
                "databases": {},
                "models": {}
            }
    
    def _save_status(self, status):
        """Save the sync status"""
        try:
            os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
            with open(self.status_path, 'w') as f:
                json.dump(status, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving sync status: {e}")
            return False
    
    def start_sync(self, system_id=None):
        """Mark the start of a synchronization"""
        system_id = system_id or "desktop"
        
        # Update status
        self.status["sync_in_progress"] = True
        
        # Add system if not exists
        if system_id not in self.status["systems"]:
            self.status["systems"][system_id] = {
                "last_sync_start": int(time.time()),
                "last_sync_complete": 0,
                "sync_count": 0
            }
        else:
            self.status["systems"][system_id]["last_sync_start"] = int(time.time())
        
        # Save status
        self._save_status(self.status)
        
        logger.info(f"Started sync for system: {system_id}")
        return True
    
    def complete_sync(self, system_id=None, success=True, details=None):
        """Mark the completion of a synchronization"""
        system_id = system_id or "desktop"
        
        # Update status
        self.status["sync_in_progress"] = False
        self.status["last_sync"] = int(time.time())
        
        # Update system status
        if system_id in self.status["systems"]:
            self.status["systems"][system_id]["last_sync_complete"] = int(time.time())
            self.status["systems"][system_id]["sync_count"] += 1
            self.status["systems"][system_id]["last_sync_success"] = success
            
            if details:
                self.status["systems"][system_id]["last_sync_details"] = details
        
        # Save status
        self._save_status(self.status)
        
        logger.info(f"Completed sync for system: {system_id} (success: {success})")
        return True
    
    def update_database_sync(self, db_id, status_info):
        """Update the sync status for a database"""
        # Add database if not exists
        if db_id not in self.status["databases"]:
            self.status["databases"][db_id] = {}
        
        # Update database status
        self.status["databases"][db_id].update(status_info)
        
        # Save status
        self._save_status(self.status)
        
        logger.info(f"Updated sync status for database: {db_id}")
        return True
    
    def update_model_sync(self, model_id, status_info):
        """Update the sync status for a model"""
        # Add model if not exists
        if model_id not in self.status["models"]:
            self.status["models"][model_id] = {}
        
        # Update model status
        self.status["models"][model_id].update(status_info)
        
        # Save status
        self._save_status(self.status)
        
        logger.info(f"Updated sync status for model: {model_id}")
        return True
    
    def get_status(self):
        """Get the current sync status"""
        return self.status
    
    def get_system_status(self, system_id=None):
        """Get the sync status for a specific system"""
        system_id = system_id or "desktop"
        return self.status["systems"].get(system_id, {})
    
    def is_sync_in_progress(self):
        """Check if a sync is in progress"""
        return self.status["sync_in_progress"]