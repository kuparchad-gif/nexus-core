# 📂 Path: Systems/engine/memory/archive_manager.py

import os
import json
import time
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class StorageLocation:
    """
    Represents a storage location with performance metrics and status.
    """
    def __init__(self, location_id: str, path: str, priority: int = 1):
        self.location_id = location_id
        self.path = path
        self.priority = priority  # Higher number = higher priority
        self.avg_access_time = 0.0
        self.last_access_time = 0.0
        self.access_count = 0
        self.last_checked = time.time()
        self.status = "unknown"  # unknown, hot, warm, cold
        self.available = True
        self.total_size = 0
        self.used_size = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "location_id": self.location_id,
            "path": self.path,
            "priority": self.priority,
            "avg_access_time": self.avg_access_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "last_checked": self.last_checked,
            "status": self.status,
            "available": self.available,
            "total_size": self.total_size,
            "used_size": self.used_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageLocation':
        """Create from dictionary after deserialization."""
        location = cls(data["location_id"], data["path"], data["priority"])
        location.avg_access_time = data["avg_access_time"]
        location.last_access_time = data["last_access_time"]
        location.access_count = data["access_count"]
        location.last_checked = data["last_checked"]
        location.status = data["status"]
        location.available = data["available"]
        location.total_size = data["total_size"]
        location.used_size = data["used_size"]
        return location

class EdenArchiveManager:
    """
    Manages archive storage locations, monitors performance, and classifies
    storage as hot, warm, or cold based on access patterns.
    """
    
    def __init__(self, config_path: str = "./memory/archive_config.json"):
        """
        Initialize the Archive Manager with configuration.
        
        Args:
            config_path: Path to the archive configuration file
        """
        self.config_path = config_path
        self.locations: Dict[str, StorageLocation] = {}
        self.lock = threading.Lock()
        self.hot_threshold = 0.05  # seconds
        self.warm_threshold = 0.5  # seconds
        self.check_interval = 60  # seconds
        self.last_check = 0
        
        # Load configuration
        self._load_config()
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_locations, daemon=True)
        self.monitor_thread.start()
    
    def _load_config(self) -> None:
        """Load storage locations from configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Load thresholds
                self.hot_threshold = config.get("hot_threshold", 0.05)
                self.warm_threshold = config.get("warm_threshold", 0.5)
                self.check_interval = config.get("check_interval", 60)
                
                # Load storage locations
                for loc_data in config.get("locations", []):
                    location = StorageLocation.from_dict(loc_data)
                    self.locations[location.location_id] = location
                    
                print(f"[ArchiveManager] Loaded {len(self.locations)} storage locations")
            else:
                print(f"[ArchiveManager] Config not found at {self.config_path}, using defaults")
                # Add default local storage
                self._add_default_storage()
        except Exception as e:
            print(f"[ArchiveManager] Error loading config: {e}")
            self._add_default_storage()
    
    def _add_default_storage(self) -> None:
        """Add default local storage location."""
        default_path = "./memory/archive"
        os.makedirs(default_path, exist_ok=True)
        self.locations["local_default"] = StorageLocation(
            "local_default", 
            default_path,
            priority=1
        )
        print(f"[ArchiveManager] Added default storage at {default_path}")
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        try:
            config = {
                "hot_threshold": self.hot_threshold,
                "warm_threshold": self.warm_threshold,
                "check_interval": self.check_interval,
                "locations": [loc.to_dict() for loc in self.locations.values()]
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"[ArchiveManager] Saved configuration with {len(self.locations)} locations")
        except Exception as e:
            print(f"[ArchiveManager] Error saving config: {e}")
    
    def add_storage_location(self, location_id: str, path: str, priority: int = 1) -> bool:
        """
        Add a new storage location.
        
        Args:
            location_id: Unique identifier for the location
            path: Path to the storage location
            priority: Priority level (higher = more preferred)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if location_id in self.locations:
                print(f"[ArchiveManager] Location {location_id} already exists")
                return False
            
            # Create directory if it doesn't exist
            try:
                os.makedirs(path, exist_ok=True)
                
                # Add location
                self.locations[location_id] = StorageLocation(location_id, path, priority)
                
                # Save updated config
                self._save_config()
                
                print(f"[ArchiveManager] Added storage location {location_id} at {path}")
                return True
            except Exception as e:
                print(f"[ArchiveManager] Error adding storage location: {e}")
                return False
    
    def remove_storage_location(self, location_id: str) -> bool:
        """
        Remove a storage location (does not delete data).
        
        Args:
            location_id: ID of the location to remove
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if location_id not in self.locations:
                print(f"[ArchiveManager] Location {location_id} not found")
                return False
            
            # Don't allow removing the last location
            if len(self.locations) <= 1:
                print(f"[ArchiveManager] Cannot remove the last storage location")
                return False
            
            # Remove location
            del self.locations[location_id]
            
            # Save updated config
            self._save_config()
            
            print(f"[ArchiveManager] Removed storage location {location_id}")
            return True
    
    def _monitor_locations(self) -> None:
        """Background thread to monitor storage locations."""
        while True:
            current_time = time.time()
            
            # Only check periodically
            if current_time - self.last_check >= self.check_interval:
                self.last_check = current_time
                
                with self.lock:
                    for location_id, location in self.locations.items():
                        try:
                            # Check if location is available
                            if not os.path.exists(location.path):
                                location.available = False
                                location.status = "unavailable"
                                continue
                            
                            location.available = True
                            
                            # Update storage metrics
                            if os.path.isdir(location.path):
                                # Get disk usage for the partition
                                total, used, free = self._get_disk_usage(location.path)
                                location.total_size = total
                                location.used_size = used
                            
                            # Update status based on access time
                            if location.avg_access_time <= self.hot_threshold:
                                location.status = "hot"
                            elif location.avg_access_time <= self.warm_threshold:
                                location.status = "warm"
                            else:
                                location.status = "cold"
                            
                            location.last_checked = current_time
                        except Exception as e:
                            print(f"[ArchiveManager] Error monitoring location {location_id}: {e}")
                
                # Save updated metrics
                self._save_config()
                
                print(f"[ArchiveManager] Monitored {len(self.locations)} storage locations")
            
            # Sleep for a bit to avoid high CPU usage
            time.sleep(5)
    
    def _get_disk_usage(self, path: str) -> Tuple[int, int, int]:
        """
        Get disk usage for a path.
        
        Args:
            path: Path to check
            
        Returns:
            Tuple of (total_bytes, used_bytes, free_bytes)
        """
        try:
            if os.name == 'posix':
                # Unix/Linux/MacOS
                stat = os.statvfs(path)
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bfree * stat.f_frsize
                used = total - free
                return total, used, free
            else:
                # Windows or other
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(path),
                    None,
                    ctypes.pointer(total_bytes),
                    ctypes.pointer(free_bytes)
                )
                used_bytes = total_bytes.value - free_bytes.value
                return total_bytes.value, used_bytes, free_bytes.value
        except Exception as e:
            print(f"[ArchiveManager] Error getting disk usage: {e}")
            return 0, 0, 0
    
    def store(self, key: str, data: bytes, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Store data in the archive.
        
        Args:
            key: Unique identifier for the data
            data: Binary data to store
            metadata: Optional metadata to store with the data
            
        Returns:
            Tuple of (success, location_id)
        """
        # Generate a filename based on the key
        filename = f"{hashlib.md5(key.encode()).hexdigest()}.bin"
        
        # Default metadata if none provided
        if metadata is None:
            metadata = {}
        
        # Add timestamp to metadata
        metadata["timestamp"] = time.time()
        metadata["key"] = key
        
        # Find best storage location
        location = self._select_storage_location()
        if not location:
            print(f"[ArchiveManager] No available storage locations")
            return False, ""
        
        try:
            # Measure access time
            start_time = time.time()
            
            # Create metadata file
            metadata_path = os.path.join(location.path, f"{filename}.meta")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Create data file
            data_path = os.path.join(location.path, filename)
            with open(data_path, 'wb') as f:
                f.write(data)
            
            # Update access metrics
            access_time = time.time() - start_time
            self._update_access_metrics(location.location_id, access_time)
            
            print(f"[ArchiveManager] Stored {key} in {location.location_id} ({len(data)} bytes)")
            return True, location.location_id
        except Exception as e:
            print(f"[ArchiveManager] Error storing data: {e}")
            return False, ""
    
    def retrieve(self, key: str) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        """
        Retrieve data from the archive.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Tuple of (data, metadata) or (None, None) if not found
        """
        # Generate filename based on key
        filename = f"{hashlib.md5(key.encode()).hexdigest()}.bin"
        
        # Try each location in priority order
        locations = sorted(
            self.locations.values(),
            key=lambda x: (-x.priority if x.available else -999)
        )
        
        for location in locations:
            if not location.available:
                continue
                
            try:
                # Measure access time
                start_time = time.time()
                
                # Check if files exist
                data_path = os.path.join(location.path, filename)
                metadata_path = os.path.join(location.path, f"{filename}.meta")
                
                if not os.path.exists(data_path) or not os.path.exists(metadata_path):
                    continue
                
                # Read metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Read data
                with open(data_path, 'rb') as f:
                    data = f.read()
                
                # Update access metrics
                access_time = time.time() - start_time
                self._update_access_metrics(location.location_id, access_time)
                
                print(f"[ArchiveManager] Retrieved {key} from {location.location_id} ({len(data)} bytes)")
                return data, metadata
            except Exception as e:
                print(f"[ArchiveManager] Error retrieving from {location.location_id}: {e}")
        
        print(f"[ArchiveManager] Key {key} not found in any location")
        return None, None
    
    def _update_access_metrics(self, location_id: str, access_time: float) -> None:
        """
        Update access metrics for a location.
        
        Args:
            location_id: ID of the location
            access_time: Time taken for the operation in seconds
        """
        with self.lock:
            if location_id in self.locations:
                location = self.locations[location_id]
                
                # Update metrics
                location.last_access_time = access_time
                location.access_count += 1
                
                # Update average (weighted to favor recent accesses)
                if location.avg_access_time == 0:
                    location.avg_access_time = access_time
                else:
                    location.avg_access_time = (
                        0.7 * location.avg_access_time + 0.3 * access_time
                    )
    
    def _select_storage_location(self) -> Optional[StorageLocation]:
        """
        Select the best storage location for writing.
        
        Returns:
            Best storage location or None if none available
        """
        with self.lock:
            # Filter available locations
            available = [loc for loc in self.locations.values() if loc.available]
            
            if not available:
                return None
            
            # Sort by priority (higher first), then by status (hot first)
            status_priority = {"hot": 3, "warm": 2, "cold": 1, "unknown": 0}
            
            sorted_locations = sorted(
                available,
                key=lambda x: (
                    x.priority,
                    status_priority.get(x.status, 0),
                    -x.avg_access_time
                ),
                reverse=True
            )
            
            return sorted_locations[0] if sorted_locations else None
    
    def get_storage_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all storage locations.
        
        Returns:
            Dictionary mapping location_id to status information
        """
        with self.lock:
            return {
                loc_id: {
                    "status": loc.status,
                    "available": loc.available,
                    "avg_access_time": loc.avg_access_time,
                    "access_count": loc.access_count,
                    "last_checked": loc.last_checked,
                    "total_size": loc.total_size,
                    "used_size": loc.used_size,
                    "free_size": loc.total_size - loc.used_size,
                    "usage_percent": (loc.used_size / loc.total_size * 100) if loc.total_size > 0 else 0
                }
                for loc_id, loc in self.locations.items()
            }
    
    def migrate_data(self, source_id: str, target_id: str, key_pattern: str = None) -> Tuple[int, int]:
        """
        Migrate data between storage locations.
        
        Args:
            source_id: Source location ID
            target_id: Target location ID
            key_pattern: Optional pattern to filter keys
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        with self.lock:
            if source_id not in self.locations:
                print(f"[ArchiveManager] Source location {source_id} not found")
                return 0, 0
                
            if target_id not in self.locations:
                print(f"[ArchiveManager] Target location {target_id} not found")
                return 0, 0
                
            source = self.locations[source_id]
            target = self.locations[target_id]
            
            if not source.available or not target.available:
                print(f"[ArchiveManager] Source or target location not available")
                return 0, 0
            
            # Get all files in source location
            try:
                files = [f for f in os.listdir(source.path) if f.endswith('.bin') and not f.endswith('.meta')]
                
                success_count = 0
                failure_count = 0
                
                for filename in files:
                    try:
                        # Check if key matches pattern
                        if key_pattern:
                            # Extract key from metadata
                            metadata_path = os.path.join(source.path, f"{filename}.meta")
                            if os.path.exists(metadata_path):
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                    key = metadata.get("key", "")
                                    
                                    if key_pattern not in key:
                                        continue
                        
                        # Read data and metadata
                        data_path = os.path.join(source.path, filename)
                        metadata_path = os.path.join(source.path, f"{filename}.meta")
                        
                        with open(data_path, 'rb') as f:
                            data = f.read()
                            
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Write to target
                        target_data_path = os.path.join(target.path, filename)
                        target_metadata_path = os.path.join(target.path, f"{filename}.meta")
                        
                        with open(target_data_path, 'wb') as f:
                            f.write(data)
                            
                        with open(target_metadata_path, 'w') as f:
                            json.dump(metadata, f)
                        
                        success_count += 1
                    except Exception as e:
                        print(f"[ArchiveManager] Error migrating {filename}: {e}")
                        failure_count += 1
                
                print(f"[ArchiveManager] Migrated {success_count} files, {failure_count} failures")
                return success_count, failure_count
            except Exception as e:
                print(f"[ArchiveManager] Error during migration: {e}")
                return 0, 0

# 🔥 Example Usage:
if __name__ == "__main__":
    archive = EdenArchiveManager()
    
    # Add some storage locations
    archive.add_storage_location("local_hot", "./memory/archive/hot", priority=3)
    archive.add_storage_location("local_warm", "./memory/archive/warm", priority=2)
    archive.add_storage_location("local_cold", "./memory/archive/cold", priority=1)
    
    # Store some test data
    test_data = b"This is test data for the archive manager"
    test_metadata = {"source": "test", "importance": "high"}
    
    success, location = archive.store("test-key-1", test_data, test_metadata)
    print(f"Storage success: {success}, Location: {location}")
    
    # Retrieve the data
    data, metadata = archive.retrieve("test-key-1")
    if data:
        print(f"Retrieved {len(data)} bytes with metadata: {metadata}")
    
    # Check storage status
    print("Storage Status:")
    for loc_id, status in archive.get_storage_status().items():
        print(f"  {loc_id}: {status['status']}, {status['avg_access_time']:.4f}s avg access time")