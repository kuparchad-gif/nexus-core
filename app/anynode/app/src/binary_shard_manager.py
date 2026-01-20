"""
Manages sharded binary memory storage for Viren/Lillith.
Stores memories as encrypted binary shards across multiple locations.
"""

import os
import uuid
import hashlib
from typing import Dict, List, Any, Optional
from ..binary_protocol import BinaryProtocol

class BinaryShardManager:
    def __init__(self, shard_locations: List[str], protocol: BinaryProtocol):
        """
        Initialize the shard manager.
        
        Args:
            shard_locations: List of directories to store shards
            protocol: Binary protocol for encoding/decoding
        """
        self.shard_locations = shard_locations
        self.protocol = protocol
        
        # Ensure shard directories exist
        for location in shard_locations:
            os.makedirs(location, exist_ok=True)
    
    def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """
        Store a memory across shards.
        
        Args:
            memory_data: Memory data to store
            
        Returns:
            Memory ID
        """
        # Generate unique ID for this memory
        memory_id = str(uuid.uuid4())
        memory_data['id'] = memory_id
        
        # Encode memory to binary
        binary_data = self.protocol.encode_thought(memory_data)
        
        # Calculate number of shards (based on data size)
        num_shards = max(3, len(binary_data) // (1024 * 1024) + 1)
        
        # Split binary data into shards
        shard_size = len(binary_data) // num_shards
        shards = [binary_data[i:i+shard_size] for i in range(0, len(binary_data), shard_size)]
        
        # Store shards across locations
        for i, shard in enumerate(shards):
            # Select location using deterministic algorithm
            location_index = int(hashlib.md5(f"{memory_id}:{i}".encode()).hexdigest(), 16) % len(self.shard_locations)
            location = self.shard_locations[location_index]
            
            # Create shard filename
            shard_filename = os.path.join(location, f"{memory_id}_shard_{i}")
            
            # Write shard to file
            with open(shard_filename, 'wb') as f:
                f.write(shard)
        
        # Store metadata about this memory
        self._store_metadata(memory_id, num_shards)
        
        return memory_id
    
    def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory from shards.
        
        Args:
            memory_id: ID of memory to retrieve
            
        Returns:
            Memory data or None if not found
        """
        # Get metadata
        metadata = self._get_metadata(memory_id)
        if not metadata:
            return None
        
        # Collect all shards
        binary_data = bytearray()
        for i in range(metadata['num_shards']):
            # Determine location using same algorithm as store
            location_index = int(hashlib.md5(f"{memory_id}:{i}".encode()).hexdigest(), 16) % len(self.shard_locations)
            location = self.shard_locations[location_index]
            
            # Read shard
            shard_filename = os.path.join(location, f"{memory_id}_shard_{i}")
            if not os.path.exists(shard_filename):
                return None  # Missing shard
            
            with open(shard_filename, 'rb') as f:
                binary_data.extend(f.read())
        
        # Decode binary data
        try:
            return self.protocol.decode_thought(binary_data)
        except Exception:
            return None
    
    def _store_metadata(self, memory_id: str, num_shards: int):
        """Store metadata about a memory"""
        metadata_dir = os.path.join(self.shard_locations[0], "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_file = os.path.join(metadata_dir, memory_id)
        with open(metadata_file, 'w') as f:
            f.write(f"num_shards={num_shards}")
    
    def _get_metadata(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a memory"""
        metadata_dir = os.path.join(self.shard_locations[0], "metadata")
        metadata_file = os.path.join(metadata_dir, memory_id)
        
        if not os.path.exists(metadata_file):
            return None
        
        metadata = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                metadata[key] = int(value) if value.isdigit() else value
        
        return metadata