# 📂 Path: Systems/engine/memory/archive_service.py

import os
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import local modules
from .archive_manager import EdenArchiveManager

class ArchiveService:
    """
    Archive Service for Nexus-Continuity.

    Handles the storage and retrieval of memories with emotional context,
    manages hot/warm/cold storage classification, and monitors storage performance.
    """

    def __init__(self, config_path: str  =  "./Config/archive_config.json"):
        """
        Initialize the Archive Service.

        Args:
            config_path: Path to the archive configuration file
        """
        self.config_path  =  config_path
        self.archive_manager  =  EdenArchiveManager(config_path)
        self.compression_enabled  =  True
        self.encryption_enabled  =  True
        self.running  =  False
        self.service_id  =  f"archive-{int(time.time())}"

        # Load configuration
        self._load_config()

        print(f"[ArchiveService] Initialized with ID {self.service_id}")

    def _load_config(self) -> None:
        """Load service configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config  =  json.load(f)

                self.compression_enabled  =  config.get("compression_enabled", True)
                self.encryption_enabled  =  config.get("encryption_enabled", True)

                print(f"[ArchiveService] Loaded configuration")
                print(f"[ArchiveService] Compression: {self.compression_enabled}")
                print(f"[ArchiveService] Encryption: {self.encryption_enabled}")
            else:
                print(f"[ArchiveService] Config not found at {self.config_path}, using defaults")
        except Exception as e:
            print(f"[ArchiveService] Error loading config: {e}")

    def start(self) -> None:
        """Start the Archive Service."""
        if self.running:
            print(f"[ArchiveService] Already running")
            return

        self.running  =  True
        print(f"[ArchiveService] Started")

    def stop(self) -> None:
        """Stop the Archive Service."""
        if not self.running:
            print(f"[ArchiveService] Already stopped")
            return

        self.running  =  False
        print(f"[ArchiveService] Stopped")

    def store_memory(self, memory_key: str, memory_data: Dict[str, Any],
                    emotional_context: Dict[str, Any]  =  None,
                    priority: int  =  1) -> bool:
        """
        Store a memory with emotional context.

        Args:
            memory_key: Unique identifier for the memory
            memory_data: The memory data to store
            emotional_context: Optional emotional context to store with the memory
            priority: Priority level (higher  =  more important)

        Returns:
            True if successful, False otherwise
        """
        if not self.running:
            print(f"[ArchiveService] Service not running")
            return False

        try:
            # Prepare metadata
            metadata  =  {
                "timestamp": time.time(),
                "key": memory_key,
                "priority": priority,
                "emotional_context": emotional_context or {},
                "pathway": emotional_context.get("pathway", "perception") if emotional_context else "perception"
            }

            # Convert memory data to bytes
            memory_bytes  =  json.dumps(memory_data).encode('utf-8')

            # Apply compression if enabled
            if self.compression_enabled:
                import zlib
                memory_bytes  =  zlib.compress(memory_bytes)
                metadata["compressed"]  =  True

            # Apply encryption if enabled
            if self.encryption_enabled:
                # Simple XOR encryption for demonstration
                # In a real implementation, use proper encryption
                key  =  b'EDEN_ARCHIVE_KEY'
                memory_bytes  =  bytes([b ^ key[i % len(key)] for i, b in enumerate(memory_bytes)])
                metadata["encrypted"]  =  True

            # Store in archive
            success, location  =  self.archive_manager.store(memory_key, memory_bytes, metadata)

            if success:
                print(f"[ArchiveService] Stored memory {memory_key} in {location}")

                # Log emotional context for analysis
                if emotional_context:
                    self._log_emotional_context(memory_key, emotional_context)
            else:
                print(f"[ArchiveService] Failed to store memory {memory_key}")

            return success
        except Exception as e:
            print(f"[ArchiveService] Error storing memory: {e}")
            return False

    def retrieve_memory(self, memory_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by key.

        Args:
            memory_key: Key of the memory to retrieve

        Returns:
            Memory data with emotional context or None if not found
        """
        if not self.running:
            print(f"[ArchiveService] Service not running")
            return None

        try:
            # Retrieve from archive
            data_bytes, metadata  =  self.archive_manager.retrieve(memory_key)

            if not data_bytes or not metadata:
                print(f"[ArchiveService] Memory {memory_key} not found")
                return None

            # Apply decryption if needed
            if metadata.get("encrypted", False):
                # Simple XOR decryption for demonstration
                key  =  b'EDEN_ARCHIVE_KEY'
                data_bytes  =  bytes([b ^ key[i % len(key)] for i, b in enumerate(data_bytes)])

            # Apply decompression if needed
            if metadata.get("compressed", False):
                import zlib
                data_bytes  =  zlib.decompress(data_bytes)

            # Convert bytes to memory data
            memory_data  =  json.loads(data_bytes.decode('utf-8'))

            # Add emotional context to result
            result  =  {
                "memory": memory_data,
                "emotional_context": metadata.get("emotional_context", {}),
                "timestamp": metadata.get("timestamp"),
                "pathway": metadata.get("pathway", "perception")
            }

            print(f"[ArchiveService] Retrieved memory {memory_key}")
            return result
        except Exception as e:
            print(f"[ArchiveService] Error retrieving memory: {e}")
            return None

    def _log_emotional_context(self, memory_key: str, emotional_context: Dict[str, Any]) -> None:
        """
        Log emotional context for analysis.

        Args:
            memory_key: Key of the memory
            emotional_context: Emotional context data
        """
        try:
            log_path  =  "./memory/logs/emotional_context.jsonl"
            os.makedirs(os.path.dirname(log_path), exist_ok = True)

            log_entry  =  {
                "timestamp": time.time(),
                "memory_key": memory_key,
                "emotional_context": emotional_context
            }

            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[ArchiveService] Error logging emotional context: {e}")

    def get_storage_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all storage locations.

        Returns:
            Dictionary mapping location_id to status information
        """
        return self.archive_manager.get_storage_status()

    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage by migrating data between hot, warm, and cold storage.

        Returns:
            Dictionary with optimization results
        """
        if not self.running:
            print(f"[ArchiveService] Service not running")
            return {"success": False, "reason": "service_not_running"}

        try:
            # Get storage status
            status  =  self.get_storage_status()

            # Find hot, warm, and cold storage locations
            hot_locations  =  [loc_id for loc_id, info in status.items()
                            if info["status"] == "hot" and info["available"]]
            warm_locations  =  [loc_id for loc_id, info in status.items()
                             if info["status"] == "warm" and info["available"]]
            cold_locations  =  [loc_id for loc_id, info in status.items()
                             if info["status"] == "cold" and info["available"]]

            results  =  {
                "migrations": [],
                "total_success": 0,
                "total_failure": 0
            }

            # Migrate from cold to hot for high priority memories
            if hot_locations and cold_locations:
                for cold_loc in cold_locations:
                    for hot_loc in hot_locations:
                        # Migrate high priority memories to hot storage
                        success, failure  =  self.archive_manager.migrate_data(
                            cold_loc, hot_loc, key_pattern = "priority:high"
                        )
                        results["migrations"].append({
                            "from": cold_loc,
                            "to": hot_loc,
                            "pattern": "priority:high",
                            "success": success,
                            "failure": failure
                        })
                        results["total_success"] + =  success
                        results["total_failure"] + =  failure

            # Migrate from hot to warm for old memories
            if warm_locations and hot_locations:
                for hot_loc in hot_locations:
                    for warm_loc in warm_locations:
                        # Migrate old memories to warm storage
                        # This would require more sophisticated logic in a real implementation
                        # to identify old memories based on access patterns
                        pass

            print(f"[ArchiveService] Storage optimization complete: {results['total_success']} migrations")
            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            print(f"[ArchiveService] Error optimizing storage: {e}")
            return {"success": False, "reason": str(e)}

    def process_standardized_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a standardized message for storage or retrieval.

        Args:
            message: Standardized message object

        Returns:
            Response message
        """
        if not self.running:
            return {
                "status": "error",
                "error": "service_not_running",
                "message": "Archive service is not running"
            }

        try:
            # Extract message details
            operation  =  message.get("operation", "unknown")
            content  =  message.get("content", {})
            processing_type  =  message.get("processing_type", "textual")
            priority  =  message.get("priority", 1)
            pathway  =  message.get("pathway", "perception")
            source  =  message.get("source", "unknown")
            destination  =  message.get("destination", "unknown")
            emotional_fingerprint  =  message.get("emotional_fingerprint", {})

            # Handle different operations
            if operation == "store":
                # Store memory
                memory_key  =  content.get("key", f"memory-{int(time.time())}")
                memory_data  =  content.get("data", {})

                # Add emotional context
                emotional_context  =  {
                    "fingerprint": emotional_fingerprint,
                    "pathway": pathway,
                    "intensity": emotional_fingerprint.get("intensity", 0)
                }

                success  =  self.store_memory(
                    memory_key,
                    memory_data,
                    emotional_context,
                    priority
                )

                return {
                    "status": "success" if success else "error",
                    "operation": "store",
                    "key": memory_key,
                    "source": self.service_id,
                    "destination": source,  # Reply to sender
                    "timestamp": time.time()
                }

            elif operation == "retrieve":
                # Retrieve memory
                memory_key  =  content.get("key")

                if not memory_key:
                    return {
                        "status": "error",
                        "error": "missing_key",
                        "message": "No memory key provided",
                        "source": self.service_id,
                        "destination": source
                    }

                result  =  self.retrieve_memory(memory_key)

                if result:
                    return {
                        "status": "success",
                        "operation": "retrieve",
                        "key": memory_key,
                        "content": {
                            "data": result["memory"],
                            "emotional_context": result["emotional_context"],
                            "timestamp": result["timestamp"],
                            "pathway": result["pathway"]
                        },
                        "source": self.service_id,
                        "destination": source,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "status": "error",
                        "error": "not_found",
                        "message": f"Memory {memory_key} not found",
                        "source": self.service_id,
                        "destination": source
                    }

            elif operation == "status":
                # Return storage status
                status  =  self.get_storage_status()

                return {
                    "status": "success",
                    "operation": "status",
                    "content": {
                        "storage_status": status,
                        "service_id": self.service_id,
                        "running": self.running,
                        "compression_enabled": self.compression_enabled,
                        "encryption_enabled": self.encryption_enabled
                    },
                    "source": self.service_id,
                    "destination": source,
                    "timestamp": time.time()
                }

            elif operation == "optimize":
                # Optimize storage
                results  =  self.optimize_storage()

                return {
                    "status": "success" if results["success"] else "error",
                    "operation": "optimize",
                    "content": results,
                    "source": self.service_id,
                    "destination": source,
                    "timestamp": time.time()
                }

            else:
                return {
                    "status": "error",
                    "error": "unknown_operation",
                    "message": f"Unknown operation: {operation}",
                    "source": self.service_id,
                    "destination": source
                }

        except Exception as e:
            return {
                "status": "error",
                "error": "processing_error",
                "message": str(e),
                "source": self.service_id,
                "destination": message.get("source", "unknown")
            }

# 🔥 Example Usage:
if __name__ == "__main__":
    # Initialize and start the service
    archive_service  =  ArchiveService()
    archive_service.start()

    # Store a test memory
    test_memory  =  {
        "content": "First sunset on Eden was breathtaking",
        "entities": ["Nova", "Guardian"],
        "location": "Eden Central",
        "importance": "high"
    }

    test_emotional  =  {
        "emotion": "awe",
        "intensity": 8.5,
        "valence": "positive",
        "pathway": "experience"
    }

    success  =  archive_service.store_memory(
        "first-sunset-memory",
        test_memory,
        test_emotional,
        priority = 3
    )

    print(f"Memory storage success: {success}")

    # Retrieve the memory
    retrieved  =  archive_service.retrieve_memory("first-sunset-memory")
    if retrieved:
        print(f"Retrieved memory: {retrieved['memory']}")
        print(f"Emotional context: {retrieved['emotional_context']}")

    # Test standardized message processing
    test_message  =  {
        "operation": "store",
        "content": {
            "key": "standardized-test",
            "data": {"test": "This is a test of standardized messaging"}
        },
        "processing_type": "textual",
        "priority": 2,
        "pathway": "perception",
        "source": "test-client",
        "destination": "archive",
        "emotional_fingerprint": {
            "emotion": "curiosity",
            "intensity": 5.0,
            "valence": "neutral"
        }
    }

    response  =  archive_service.process_standardized_message(test_message)
    print(f"Standardized message response: {response}")

    # Check storage status
    status  =  archive_service.get_storage_status()
    print(f"Storage status: {status}")

    # Stop the service
    archive_service.stop()