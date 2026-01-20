#!/usr/bin/env python
"""
Memory Consolidation - Transfers short-term memories to long-term storage
"""

import os
import json
import time
import pickle
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

class MemoryType(Enum):
    """Types of memory in the system"""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryPriority(Enum):
    """Priority levels for memory retention"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class MemoryItem:
    """A single memory item with metadata"""
    
    def __init__(self, 
                content: Any, 
                memory_type: MemoryType = MemoryType.SHORT_TERM,
                priority: MemoryPriority = MemoryPriority.MEDIUM,
                tags: List[str] = None,
                metadata: Dict[str, Any] = None):
        """Initialize a memory item
        
        Args:
            content: The memory content
            memory_type: Type of memory
            priority: Priority level for retention
            tags: List of tags for categorization
            metadata: Additional metadata
        """
        self.id = f"mem_{int(time.time())}_{id(self)}"
        self.content = content
        self.memory_type = memory_type
        self.priority = priority
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.consolidation_count = 0
        self.emotional_context = {}
    
    def access(self):
        """Record memory access"""
        self.last_accessed = time.time()
        self.access_count += 1
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "consolidation_count": self.consolidation_count,
            "emotional_context": self.emotional_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary representation"""
        memory = cls(
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            priority=MemoryPriority(data["priority"]),
            tags=data["tags"],
            metadata=data["metadata"]
        )
        memory.id = data["id"]
        memory.created_at = data["created_at"]
        memory.last_accessed = data["last_accessed"]
        memory.access_count = data["access_count"]
        memory.consolidation_count = data["consolidation_count"]
        memory.emotional_context = data.get("emotional_context", {})
        return memory

class MemoryConsolidation:
    """Memory consolidation system that transfers memories between storage types"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the memory consolidation system
        
        Args:
            storage_path: Path to store memory data
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "memory")
        
        # Create storage directories
        self.short_term_path = os.path.join(self.storage_path, "short_term")
        self.working_path = os.path.join(self.storage_path, "working")
        self.long_term_path = os.path.join(self.storage_path, "long_term")
        
        os.makedirs(self.short_term_path, exist_ok=True)
        os.makedirs(self.working_path, exist_ok=True)
        os.makedirs(self.long_term_path, exist_ok=True)
        
        # Memory stores
        self.short_term_memory = {}
        self.working_memory = {}
        self.long_term_memory = {}
        
        # Consolidation settings
        self.consolidation_interval = 60  # seconds
        self.short_term_retention = 3600  # 1 hour
        self.working_retention = 86400  # 24 hours
        
        # Importance thresholds
        self.importance_threshold = 0.5  # Minimum importance to transfer to long-term
        
        # Statistics
        self.stats = {
            "short_term_count": 0,
            "working_count": 0,
            "long_term_count": 0,
            "consolidations": 0,
            "last_consolidation": None
        }
        
        # Load existing memories
        self._load_memories()
        
        # Start consolidation thread
        self.running = True
        self.consolidation_thread = threading.Thread(target=self._consolidation_loop)
        self.consolidation_thread.daemon = True
        self.consolidation_thread.start()
    
    def _load_memories(self):
        """Load existing memories from storage"""
        # Load short-term memories
        self._load_memory_store(self.short_term_path, self.short_term_memory)
        self.stats["short_term_count"] = len(self.short_term_memory)
        
        # Load working memories
        self._load_memory_store(self.working_path, self.working_memory)
        self.stats["working_count"] = len(self.working_memory)
        
        # Load long-term memories
        self._load_memory_store(self.long_term_path, self.long_term_memory)
        self.stats["long_term_count"] = len(self.long_term_memory)
        
        print(f"Loaded {self.stats['short_term_count']} short-term, "
              f"{self.stats['working_count']} working, and "
              f"{self.stats['long_term_count']} long-term memories")
    
    def _load_memory_store(self, path: str, store: Dict[str, MemoryItem]):
        """Load memories from a specific path into a store"""
        if not os.path.exists(path):
            return
        
        memory_files = [f for f in os.listdir(path) if f.endswith('.json')]
        for memory_file in memory_files:
            try:
                with open(os.path.join(path, memory_file), 'r') as f:
                    data = json.load(f)
                    memory = MemoryItem.from_dict(data)
                    store[memory.id] = memory
            except Exception as e:
                print(f"Error loading memory {memory_file}: {e}")
    
    def _save_memory(self, memory: MemoryItem):
        """Save a memory item to the appropriate storage"""
        try:
            # Determine storage path based on memory type
            if memory.memory_type == MemoryType.SHORT_TERM:
                path = os.path.join(self.short_term_path, f"{memory.id}.json")
            elif memory.memory_type == MemoryType.WORKING:
                path = os.path.join(self.working_path, f"{memory.id}.json")
            else:
                path = os.path.join(self.long_term_path, f"{memory.id}.json")
            
            # Save memory
            with open(path, 'w') as f:
                json.dump(memory.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving memory {memory.id}: {e}")
            return False
    
    def _delete_memory_file(self, memory: MemoryItem):
        """Delete a memory file from storage"""
        try:
            # Determine storage path based on memory type
            if memory.memory_type == MemoryType.SHORT_TERM:
                path = os.path.join(self.short_term_path, f"{memory.id}.json")
            elif memory.memory_type == MemoryType.WORKING:
                path = os.path.join(self.working_path, f"{memory.id}.json")
            else:
                path = os.path.join(self.long_term_path, f"{memory.id}.json")
            
            # Delete file if it exists
            if os.path.exists(path):
                os.remove(path)
            
            return True
        except Exception as e:
            print(f"Error deleting memory file {memory.id}: {e}")
            return False
    
    def _consolidation_loop(self):
        """Background thread for memory consolidation"""
        while self.running:
            try:
                self.consolidate_memories()
            except Exception as e:
                print(f"Error in consolidation loop: {e}")
            
            # Sleep until next consolidation
            time.sleep(self.consolidation_interval)
    
    def add_memory(self, 
                  content: Any, 
                  memory_type: MemoryType = MemoryType.SHORT_TERM,
                  priority: MemoryPriority = MemoryPriority.MEDIUM,
                  tags: List[str] = None,
                  metadata: Dict[str, Any] = None,
                  emotional_context: Dict[str, float] = None) -> str:
        """Add a new memory
        
        Args:
            content: Memory content
            memory_type: Type of memory
            priority: Priority level
            tags: List of tags
            metadata: Additional metadata
            emotional_context: Emotional context values
            
        Returns:
            Memory ID
        """
        # Create memory item
        memory = MemoryItem(
            content=content,
            memory_type=memory_type,
            priority=priority,
            tags=tags,
            metadata=metadata
        )
        
        # Add emotional context if provided
        if emotional_context:
            memory.emotional_context = emotional_context
        
        # Store in appropriate memory store
        if memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory[memory.id] = memory
            self.stats["short_term_count"] += 1
        elif memory_type == MemoryType.WORKING:
            self.working_memory[memory.id] = memory
            self.stats["working_count"] += 1
        else:
            self.long_term_memory[memory.id] = memory
            self.stats["long_term_count"] += 1
        
        # Save to storage
        self._save_memory(memory)
        
        return memory.id
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory item or None if not found
        """
        # Check each memory store
        if memory_id in self.short_term_memory:
            memory = self.short_term_memory[memory_id]
            memory.access()
            return memory
        elif memory_id in self.working_memory:
            memory = self.working_memory[memory_id]
            memory.access()
            return memory
        elif memory_id in self.long_term_memory:
            memory = self.long_term_memory[memory_id]
            memory.access()
            return memory
        
        return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check each memory store
        if memory_id in self.short_term_memory:
            memory = self.short_term_memory[memory_id]
            del self.short_term_memory[memory_id]
            self.stats["short_term_count"] -= 1
            self._delete_memory_file(memory)
            return True
        elif memory_id in self.working_memory:
            memory = self.working_memory[memory_id]
            del self.working_memory[memory_id]
            self.stats["working_count"] -= 1
            self._delete_memory_file(memory)
            return True
        elif memory_id in self.long_term_memory:
            memory = self.long_term_memory[memory_id]
            del self.long_term_memory[memory_id]
            self.stats["long_term_count"] -= 1
            self._delete_memory_file(memory)
            return True
        
        return False
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """Perform memory consolidation
        
        Returns:
            Dictionary with consolidation results
        """
        now = time.time()
        short_to_working = 0
        working_to_long = 0
        short_to_deleted = 0
        working_to_deleted = 0
        
        # Process short-term to working memory
        short_term_ids = list(self.short_term_memory.keys())
        for memory_id in short_term_ids:
            if memory_id not in self.short_term_memory:
                continue
                
            memory = self.short_term_memory[memory_id]
            age = now - memory.created_at
            
            # Check if memory should be consolidated
            if age > self.short_term_retention:
                # Calculate importance score
                importance = self._calculate_importance(memory)
                
                # If important enough, move to working memory
                if importance > 0 or memory.priority.value >= MemoryPriority.HIGH.value:
                    # Update memory type
                    memory.memory_type = MemoryType.WORKING
                    memory.consolidation_count += 1
                    
                    # Move to working memory
                    self.working_memory[memory_id] = memory
                    del self.short_term_memory[memory_id]
                    
                    # Update stats
                    self.stats["short_term_count"] -= 1
                    self.stats["working_count"] += 1
                    short_to_working += 1
                    
                    # Save to new location
                    self._save_memory(memory)
                    self._delete_memory_file(memory)
                else:
                    # Not important enough, delete
                    del self.short_term_memory[memory_id]
                    self.stats["short_term_count"] -= 1
                    self._delete_memory_file(memory)
                    short_to_deleted += 1
        
        # Process working to long-term memory
        working_ids = list(self.working_memory.keys())
        for memory_id in working_ids:
            if memory_id not in self.working_memory:
                continue
                
            memory = self.working_memory[memory_id]
            age = now - memory.created_at
            
            # Check if memory should be consolidated
            if age > self.working_retention:
                # Calculate importance score
                importance = self._calculate_importance(memory)
                
                # If important enough, move to long-term memory
                if importance > self.importance_threshold or memory.priority.value >= MemoryPriority.HIGH.value:
                    # Update memory type
                    memory.memory_type = MemoryType.LONG_TERM
                    memory.consolidation_count += 1
                    
                    # Move to long-term memory
                    self.long_term_memory[memory_id] = memory
                    del self.working_memory[memory_id]
                    
                    # Update stats
                    self.stats["working_count"] -= 1
                    self.stats["long_term_count"] += 1
                    working_to_long += 1
                    
                    # Save to new location
                    self._save_memory(memory)
                    self._delete_memory_file(memory)
                else:
                    # Not important enough, delete
                    del self.working_memory[memory_id]
                    self.stats["working_count"] -= 1
                    self._delete_memory_file(memory)
                    working_to_deleted += 1
        
        # Update consolidation stats
        self.stats["consolidations"] += 1
        self.stats["last_consolidation"] = now
        
        return {
            "timestamp": now,
            "short_to_working": short_to_working,
            "working_to_long": working_to_long,
            "short_to_deleted": short_to_deleted,
            "working_to_deleted": working_to_deleted,
            "total_processed": short_to_working + working_to_long + short_to_deleted + working_to_deleted
        }
    
    def _calculate_importance(self, memory: MemoryItem) -> float:
        """Calculate importance score for a memory
        
        Args:
            memory: Memory item
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        # Base importance from priority
        base_importance = memory.priority.value / MemoryPriority.CRITICAL.value
        
        # Adjust based on access count and recency
        access_factor = min(1.0, memory.access_count / 5.0)  # Max out at 5 accesses
        
        # Recency factor (higher if accessed recently)
        time_since_access = time.time() - memory.last_accessed
        recency_factor = max(0.0, 1.0 - (time_since_access / (24 * 3600)))  # Decay over 24 hours
        
        # Emotional intensity factor
        emotional_intensity = 0.0
        if memory.emotional_context:
            # Use the maximum emotional value as intensity
            emotional_intensity = max(memory.emotional_context.values()) if memory.emotional_context.values() else 0.0
        
        # Combine factors with weights
        importance = (
            0.4 * base_importance +
            0.3 * access_factor +
            0.2 * recency_factor +
            0.1 * emotional_intensity
        )
        
        return importance
    
    def search_memories(self, 
                       query: str = None, 
                       tags: List[str] = None,
                       memory_type: MemoryType = None,
                       min_priority: MemoryPriority = None,
                       limit: int = 10) -> List[MemoryItem]:
        """Search for memories
        
        Args:
            query: Search query
            tags: Filter by tags
            memory_type: Filter by memory type
            min_priority: Minimum priority
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        results = []
        
        # Collect memories from all stores
        all_memories = list(self.short_term_memory.values())
        all_memories.extend(self.working_memory.values())
        all_memories.extend(self.long_term_memory.values())
        
        # Apply filters
        for memory in all_memories:
            # Filter by memory type
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Filter by minimum priority
            if min_priority and memory.priority.value < min_priority.value:
                continue
            
            # Filter by tags
            if tags and not all(tag in memory.tags for tag in tags):
                continue
            
            # Filter by query
            if query:
                # Simple string matching for now
                content_str = str(memory.content)
                if query.lower() not in content_str.lower():
                    continue
            
            # Add to results
            results.append(memory)
            
            # Check limit
            if len(results) >= limit:
                break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.stats
    
    def stop(self):
        """Stop the consolidation thread"""
        self.running = False
        if self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=1.0)

# Example usage
if __name__ == "__main__":
    # Create memory consolidation system
    memory_system = MemoryConsolidation()
    
    # Add some test memories
    memory_system.add_memory(
        content="This is a short-term memory",
        memory_type=MemoryType.SHORT_TERM,
        priority=MemoryPriority.MEDIUM,
        tags=["test", "example"],
        emotional_context={"joy": 0.7}
    )
    
    memory_system.add_memory(
        content="This is a working memory",
        memory_type=MemoryType.WORKING,
        priority=MemoryPriority.HIGH,
        tags=["test", "important"]
    )
    
    memory_system.add_memory(
        content="This is a long-term memory",
        memory_type=MemoryType.LONG_TERM,
        priority=MemoryPriority.CRITICAL,
        tags=["test", "permanent"]
    )
    
    # Search for memories
    results = memory_system.search_memories(query="memory", tags=["test"])
    print(f"Search results: {len(results)} memories found")
    for memory in results:
        print(f"- {memory.id}: {memory.content} ({memory.memory_type.value})")
    
    # Get statistics
    stats = memory_system.get_stats()
    print(f"Memory stats: {stats}")
    
    # Manually trigger consolidation
    consolidation_results = memory_system.consolidate_memories()
    print(f"Consolidation results: {consolidation_results}")
    
    # Stop the system
    memory_system.stop()