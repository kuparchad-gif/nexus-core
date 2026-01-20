#!/usr/bin/env python3
"""
Database Manager for Cloud Viren
Provides database access for all Viren components
"""

import os
import sys
import json
import time
import logging
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VirenDatabaseManager")

class DatabaseManager:
    """
    Database Manager for Cloud Viren
    Provides database access for all Viren components
    """
    
    def __init__(self, db_path: str = None, config_path: str = None):
        """Initialize the database manager"""
        self.config_path = config_path or os.path.join("config", "database_config.json")
        self.config = self._load_config()
        self.db_path = db_path or self.config.get("db_path", os.path.join("data", "viren.db"))
        self.connection = None
        self.cursor = None
        self.lock = threading.RLock()
        self.initialized_tables = set()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        self._connect()
        
        # Initialize tables
        self._initialize_tables()
        
        logger.info(f"Database manager initialized with database at {self.db_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "db_path": os.path.join("data", "viren.db"),
            "backup_interval": 86400,  # 24 hours
            "max_backups": 5,
            "vacuum_interval": 604800,  # 7 days
            "tables": {
                "models": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "model_id TEXT NOT NULL",
                        "model_size TEXT",
                        "model_type TEXT",
                        "provider TEXT",
                        "status TEXT",
                        "downloaded INTEGER",
                        "last_check REAL",
                        "created_at REAL",
                        "updated_at REAL"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_models_model_id ON models(model_id)",
                        "CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)"
                    ]
                },
                "diagnostics": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "device_id TEXT NOT NULL",
                        "component TEXT NOT NULL",
                        "status TEXT",
                        "data TEXT",
                        "timestamp REAL",
                        "synced INTEGER DEFAULT 0"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_diagnostics_device_id ON diagnostics(device_id)",
                        "CREATE INDEX IF NOT EXISTS idx_diagnostics_component ON diagnostics(component)",
                        "CREATE INDEX IF NOT EXISTS idx_diagnostics_timestamp ON diagnostics(timestamp)"
                    ]
                },
                "research": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "query TEXT NOT NULL",
                        "context TEXT",
                        "tentacles_used TEXT",
                        "findings TEXT",
                        "timestamp REAL",
                        "synced INTEGER DEFAULT 0"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_research_timestamp ON research(timestamp)"
                    ]
                },
                "blockchain": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "transaction_id TEXT NOT NULL",
                        "transaction_type TEXT NOT NULL",
                        "data TEXT",
                        "signature TEXT",
                        "timestamp REAL",
                        "status TEXT",
                        "synced INTEGER DEFAULT 0"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_blockchain_transaction_id ON blockchain(transaction_id)",
                        "CREATE INDEX IF NOT EXISTS idx_blockchain_timestamp ON blockchain(timestamp)"
                    ]
                },
                "devices": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "device_id TEXT NOT NULL",
                        "device_name TEXT",
                        "platform TEXT",
                        "version TEXT",
                        "capabilities TEXT",
                        "last_connection REAL",
                        "status TEXT",
                        "created_at REAL",
                        "updated_at REAL"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_devices_device_id ON devices(device_id)",
                        "CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status)"
                    ]
                },
                "alerts": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "level TEXT NOT NULL",
                        "component TEXT NOT NULL",
                        "message TEXT NOT NULL",
                        "data TEXT",
                        "timestamp REAL",
                        "acknowledged INTEGER DEFAULT 0",
                        "synced INTEGER DEFAULT 0"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level)",
                        "CREATE INDEX IF NOT EXISTS idx_alerts_component ON alerts(component)",
                        "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)"
                    ]
                },
                "sync_queue": {
                    "columns": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "item_type TEXT NOT NULL",
                        "data TEXT NOT NULL",
                        "timestamp REAL",
                        "attempts INTEGER DEFAULT 0",
                        "last_attempt REAL",
                        "status TEXT DEFAULT 'pending'"
                    ],
                    "indexes": [
                        "CREATE INDEX IF NOT EXISTS idx_sync_queue_item_type ON sync_queue(item_type)",
                        "CREATE INDEX IF NOT EXISTS idx_sync_queue_status ON sync_queue(status)"
                    ]
                }
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config.get(key), dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    
                    logger.info("Database configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading database configuration: {e}")
        
        logger.info("Using default database configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Database configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving database configuration: {e}")
            return False
    
    def _connect(self) -> None:
        """Connect to the database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _initialize_tables(self) -> None:
        """Initialize database tables"""
        with self.lock:
            for table_name, table_info in self.config["tables"].items():
                self._initialize_table(table_name, table_info)
    
    def _initialize_table(self, table_name: str, table_info: Dict[str, Any]) -> None:
        """Initialize a specific database table"""
        try:
            # Create table if it doesn't exist
            columns = ", ".join(table_info["columns"])
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
            self.cursor.execute(create_table_sql)
            
            # Create indexes
            for index_sql in table_info.get("indexes", []):
                self.cursor.execute(index_sql)
            
            self.connection.commit()
            self.initialized_tables.add(table_name)
            logger.info(f"Initialized table: {table_name}")
        
        except Exception as e:
            logger.error(f"Error initializing table {table_name}: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL query"""
        with self.lock:
            try:
                return self.cursor.execute(sql, params)
            except Exception as e:
                logger.error(f"Error executing SQL: {sql}, params: {params}, error: {e}")
                raise
    
    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """Execute a SQL query with multiple parameter sets"""
        with self.lock:
            try:
                return self.cursor.executemany(sql, params_list)
            except Exception as e:
                logger.error(f"Error executing SQL: {sql}, params_list: {params_list}, error: {e}")
                raise
    
    def commit(self) -> None:
        """Commit changes to the database"""
        with self.lock:
            try:
                self.connection.commit()
            except Exception as e:
                logger.error(f"Error committing changes: {e}")
                raise
    
    def rollback(self) -> None:
        """Rollback changes to the database"""
        with self.lock:
            try:
                self.connection.rollback()
            except Exception as e:
                logger.error(f"Error rolling back changes: {e}")
                raise
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert data into a table"""
        if table not in self.initialized_tables:
            raise ValueError(f"Table {table} not initialized")
        
        with self.lock:
            try:
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?"] * len(data))
                values = tuple(data.values())
                
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                self.cursor.execute(sql, values)
                self.connection.commit()
                
                return self.cursor.lastrowid
            except Exception as e:
                logger.error(f"Error inserting into {table}: {e}")
                self.connection.rollback()
                raise
    
    def update(self, table: str, data: Dict[str, Any], where: str, where_params: tuple) -> int:
        """Update data in a table"""
        if table not in self.initialized_tables:
            raise ValueError(f"Table {table} not initialized")
        
        with self.lock:
            try:
                set_clause = ", ".join([f"{column} = ?" for column in data.keys()])
                values = tuple(data.values()) + where_params
                
                sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
                self.cursor.execute(sql, values)
                self.connection.commit()
                
                return self.cursor.rowcount
            except Exception as e:
                logger.error(f"Error updating {table}: {e}")
                self.connection.rollback()
                raise
    
    def delete(self, table: str, where: str, where_params: tuple) -> int:
        """Delete data from a table"""
        if table not in self.initialized_tables:
            raise ValueError(f"Table {table} not initialized")
        
        with self.lock:
            try:
                sql = f"DELETE FROM {table} WHERE {where}"
                self.cursor.execute(sql, where_params)
                self.connection.commit()
                
                return self.cursor.rowcount
            except Exception as e:
                logger.error(f"Error deleting from {table}: {e}")
                self.connection.rollback()
                raise
    
    def select(self, table: str, columns: str = "*", where: str = None, where_params: tuple = None,
              order_by: str = None, limit: int = None, offset: int = None) -> List[Dict[str, Any]]:
        """Select data from a table"""
        if table not in self.initialized_tables:
            raise ValueError(f"Table {table} not initialized")
        
        with self.lock:
            try:
                sql = f"SELECT {columns} FROM {table}"
                
                params = []
                if where:
                    sql += f" WHERE {where}"
                    if where_params:
                        params.extend(where_params)
                
                if order_by:
                    sql += f" ORDER BY {order_by}"
                
                if limit is not None:
                    sql += f" LIMIT {limit}"
                
                if offset is not None:
                    sql += f" OFFSET {offset}"
                
                self.cursor.execute(sql, params)
                rows = self.cursor.fetchall()
                
                # Convert rows to dictionaries
                result = []
                for row in rows:
                    result.append({key: row[key] for key in row.keys()})
                
                return result
            except Exception as e:
                logger.error(f"Error selecting from {table}: {e}")
                raise
    
    def count(self, table: str, where: str = None, where_params: tuple = None) -> int:
        """Count rows in a table"""
        if table not in self.initialized_tables:
            raise ValueError(f"Table {table} not initialized")
        
        with self.lock:
            try:
                sql = f"SELECT COUNT(*) as count FROM {table}"
                
                params = []
                if where:
                    sql += f" WHERE {where}"
                    if where_params:
                        params.extend(where_params)
                
                self.cursor.execute(sql, params)
                result = self.cursor.fetchone()
                
                return result["count"] if result else 0
            except Exception as e:
                logger.error(f"Error counting rows in {table}: {e}")
                raise
    
    def backup(self, backup_path: str = None) -> bool:
        """Backup the database"""
        if not backup_path:
            backup_dir = os.path.join("data", "backups")
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"viren_{timestamp}.db")
        
        with self.lock:
            try:
                # Create a new connection to the backup file
                backup_conn = sqlite3.connect(backup_path)
                
                # Backup the database
                self.connection.backup(backup_conn)
                
                # Close the backup connection
                backup_conn.close()
                
                logger.info(f"Database backed up to {backup_path}")
                
                # Clean up old backups
                self._cleanup_backups()
                
                return True
            except Exception as e:
                logger.error(f"Error backing up database: {e}")
                return False
    
    def _cleanup_backups(self) -> None:
        """Clean up old backups"""
        try:
            backup_dir = os.path.join("data", "backups")
            if not os.path.exists(backup_dir):
                return
            
            # Get list of backup files
            backup_files = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.startswith("viren_") and f.endswith(".db")]
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove excess backups
            max_backups = self.config.get("max_backups", 5)
            if len(backup_files) > max_backups:
                for old_backup in backup_files[max_backups:]:
                    os.remove(old_backup)
                    logger.info(f"Removed old backup: {old_backup}")
        
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    def vacuum(self) -> bool:
        """Vacuum the database to optimize it"""
        with self.lock:
            try:
                self.cursor.execute("VACUUM")
                logger.info("Database vacuumed successfully")
                return True
            except Exception as e:
                logger.error(f"Error vacuuming database: {e}")
                return False
    
    def get_table_info(self, table: str) -> List[Dict[str, Any]]:
        """Get information about a table"""
        if table not in self.initialized_tables:
            raise ValueError(f"Table {table} not initialized")
        
        with self.lock:
            try:
                self.cursor.execute(f"PRAGMA table_info({table})")
                rows = self.cursor.fetchall()
                
                # Convert rows to dictionaries
                result = []
                for row in rows:
                    result.append({key: row[key] for key in row.keys()})
                
                return result
            except Exception as e:
                logger.error(f"Error getting table info for {table}: {e}")
                raise
    
    def get_database_size(self) -> int:
        """Get the size of the database file in bytes"""
        try:
            return os.path.getsize(self.db_path)
        except Exception as e:
            logger.error(f"Error getting database size: {e}")
            return 0
    
    def get_row_counts(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        result = {}
        
        for table in self.initialized_tables:
            try:
                result[table] = self.count(table)
            except Exception as e:
                logger.error(f"Error getting row count for {table}: {e}")
                result[table] = -1
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get database status"""
        return {
            "path": self.db_path,
            "size": self.get_database_size(),
            "tables": len(self.initialized_tables),
            "row_counts": self.get_row_counts(),
            "initialized_tables": list(self.initialized_tables)
        }

# Example usage
if __name__ == "__main__":
    # Create database manager
    db = DatabaseManager()
    
    # Insert some test data
    device_id = db.insert("devices", {
        "device_id": "test-device-1",
        "device_name": "Test Device",
        "platform": "windows",
        "version": "1.0.0",
        "capabilities": json.dumps({"models": ["3B", "7B"], "diagnostics": True}),
        "last_connection": time.time(),
        "status": "active",
        "created_at": time.time(),
        "updated_at": time.time()
    })
    
    print(f"Inserted device with ID: {device_id}")
    
    # Select data
    devices = db.select("devices", where="device_id = ?", where_params=("test-device-1",))
    print(f"Selected devices: {devices}")
    
    # Update data
    updated = db.update("devices", 
                      {"status": "offline", "updated_at": time.time()},
                      "device_id = ?", ("test-device-1",))
    print(f"Updated {updated} rows")
    
    # Get database status
    status = db.get_status()
    print(f"Database status: {status}")
    
    # Close database
    db.close()