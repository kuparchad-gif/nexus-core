#!/usr/bin/env python3
"""
Database Registry for Viren
Tracks all databases across systems, versions, and relationships
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatabaseRegistry")

class DatabaseRegistry:
    """Registry for tracking all databases across Viren systems"""
    
    def __init__(self, config_path: str = None):
        """Initialize the database registry"""
        self.config_path = config_path or os.path.join('C:/Viren/config', 'database_registry.json')
        self.registry = {}
        self.system_id = self._get_system_id()
        self._load_registry()
    
    def _get_system_id(self) -> str:
        """Get or create a unique system ID"""
        system_id_path = os.path.join('C:/Viren/config', 'system_id.json')
        
        try:
            if os.path.exists(system_id_path):
                with open(system_id_path, 'r') as f:
                    data = json.load(f)
                    return data.get('system_id', str(uuid.uuid4()))
            else:
                system_id = str(uuid.uuid4())
                os.makedirs(os.path.dirname(system_id_path), exist_ok=True)
                with open(system_id_path, 'w') as f:
                    json.dump({'system_id': system_id}, f)
                return system_id
        except Exception as e:
            logger.error(f"Error getting system ID: {e}")
            return str(uuid.uuid4())
    
    def _load_registry(self) -> None:
        """Load the database registry"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.registry = json.load(f)
            else:
                self.registry = {
                    "systems": {},
                    "databases": {},
                    "relationships": {},
                    "last_updated": time.time()
                }
                self._save_registry()
        except Exception as e:
            logger.error(f"Error loading database registry: {e}")
            self.registry = {
                "systems": {},
                "databases": {},
                "relationships": {},
                "last_updated": time.time()
            }
    
    def _save_registry(self) -> bool:
        """Save the database registry"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving database registry: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize the database registry"""
        # Register this system if not already registered
        if self.system_id not in self.registry["systems"]:
            self.registry["systems"][self.system_id] = {
                "name": os.environ.get("VIREN_SYSTEM_NAME", "Viren-" + self.system_id[:8]),
                "type": os.environ.get("VIREN_SYSTEM_TYPE", "desktop"),
                "last_seen": time.time()
            }
            self._save_registry()
        
        # Update last seen timestamp
        self.registry["systems"][self.system_id]["last_seen"] = time.time()
        self._save_registry()
        
        return True
    
    def register_database(self, db_id: str, db_info: Dict[str, Any]) -> bool:
        """Register a database in the registry"""
        if db_id not in self.registry["databases"]:
            self.registry["databases"][db_id] = {
                "system_id": self.system_id,
                "created_at": time.time(),
                "updated_at": time.time(),
                **db_info
            }
        else:
            self.registry["databases"][db_id].update({
                "updated_at": time.time(),
                **db_info
            })
        
        self.registry["last_updated"] = time.time()
        return self._save_registry()
    
    def register_relationship(self, source_db_id: str, target_db_id: str, relationship_type: str) -> bool:
        """Register a relationship between databases"""
        if source_db_id not in self.registry["databases"] or target_db_id not in self.registry["databases"]:
            logger.error(f"Cannot register relationship: one or both databases not found")
            return False
        
        relationship_id = f"{source_db_id}:{target_db_id}"
        
        self.registry["relationships"][relationship_id] = {
            "source_db_id": source_db_id,
            "target_db_id": target_db_id,
            "type": relationship_type,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        self.registry["last_updated"] = time.time()
        return self._save_registry()
    
    def get_database(self, db_id: str) -> Optional[Dict[str, Any]]:
        """Get database information by ID"""
        return self.registry["databases"].get(db_id)
    
    def get_system_databases(self, system_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Get all databases for a system"""
        system_id = system_id or self.system_id
        
        return {
            db_id: db_info
            for db_id, db_info in self.registry["databases"].items()
            if db_info.get("system_id") == system_id
        }
    
    def get_database_relationships(self, db_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all relationships for a database"""
        relationships = {}
        
        for rel_id, rel_info in self.registry["relationships"].items():
            if rel_info["source_db_id"] == db_id or rel_info["target_db_id"] == db_id:
                relationships[rel_id] = rel_info
        
        return relationships
    
    def sync_with_remote(self, remote_registry: Dict[str, Any]) -> bool:
        """Sync with a remote registry"""
        try:
            # Update systems
            for system_id, system_info in remote_registry.get("systems", {}).items():
                if system_id != self.system_id:  # Don't overwrite our own system info
                    self.registry["systems"][system_id] = system_info
            
            # Update databases
            for db_id, db_info in remote_registry.get("databases", {}).items():
                if db_id not in self.registry["databases"]:
                    self.registry["databases"][db_id] = db_info
                elif db_info.get("updated_at", 0) > self.registry["databases"][db_id].get("updated_at", 0):
                    self.registry["databases"][db_id] = db_info
            
            # Update relationships
            for rel_id, rel_info in remote_registry.get("relationships", {}).items():
                if rel_id not in self.registry["relationships"]:
                    self.registry["relationships"][rel_id] = rel_info
                elif rel_info.get("updated_at", 0) > self.registry["relationships"][rel_id].get("updated_at", 0):
                    self.registry["relationships"][rel_id] = rel_info
            
            self.registry["last_updated"] = time.time()
            return self._save_registry()
        
        except Exception as e:
            logger.error(f"Error syncing with remote registry: {e}")
            return False
    
    def get_full_registry(self) -> Dict[str, Any]:
        """Get the full registry"""
        return self.registry