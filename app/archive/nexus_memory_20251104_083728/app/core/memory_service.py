import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import os
import pickle
from datetime import datetime

class MemoryType(Enum):
    """Types of memories in the system"""
    FACTUAL = 1
    EMOTIONAL = 2
    PROCEDURAL = 3
    EPISODIC = 4
    SEMANTIC = 5

class MemoryShard:
    """A shard of memory with emotional context"""
    
    def __init__(self, 
                 content: Any, 
                 memory_type: MemoryType = MemoryType.FACTUAL,
                 emotional_tone: Dict[str, float] = None,
                 context_references: List[str] = None,
                 metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.emotional_tone = emotional_tone or {}
        self.context_references = context_references or []
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory shard to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.name,
            "emotional_tone": self.emotional_tone,
            "context_references": self.context_references,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryShard':
        """Create memory shard from dictionary"""
        shard = cls(
            content=data["content"],
            memory_type=MemoryType[data["memory_type"]],
            emotional_tone=data["emotional_tone"],
            context_references=data["context_references"],
            metadata=data["metadata"]
        )
        shard.id = data["id"]
        shard.created_at = data["created_at"]
        shard.last_accessed = data["last_accessed"]
        shard.access_count = data["access_count"]
        return shard
    
    def access(self):
        """Record memory access"""
        self.last_accessed = time.time()
        self.access_count += 1
        return self

class MemoryStore:
    """Storage for memory shards"""
    
    def __init__(self, storage_dir: str = None):
        self.memories: Dict[str, MemoryShard] = {}
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(__file__), "storage")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def add(self, shard: MemoryShard) -> str:
        """Add a memory shard to the store"""
        self.memories[shard.id] = shard
        return shard.id
    
    def get(self, shard_id: str) -> Optional[MemoryShard]:
        """Get a memory shard by ID"""
        if shard_id in self.memories:
            return self.memories[shard_id].access()
        return None
    
    def remove(self, shard_id: str) -> bool:
        """Remove a memory shard"""
        if shard_id in self.memories:
            del self.memories[shard_id]
            return True
        return False
    
    def query(self, 
              memory_type: Optional[MemoryType] = None,
              context_reference: Optional[str] = None,
              metadata_filter: Optional[Dict[str, Any]] = None,
              limit: int = 100,
              sort_by: str = "last_accessed") -> List[MemoryShard]:
        """Query memory shards"""
        results = list(self.memories.values())
        
        # Apply filters
        if memory_type:
            results = [m for m in results if m.memory_type == memory_type]
        
        if context_reference:
            results = [m for m in results if context_reference in m.context_references]
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                results = [m for m in results if key in m.metadata and m.metadata[key] == value]
        
        # Sort results
        if sort_by == "last_accessed":
            results.sort(key=lambda m: m.last_accessed, reverse=True)
        elif sort_by == "created_at":
            results.sort(key=lambda m: m.created_at, reverse=True)
        elif sort_by == "access_count":
            results.sort(key=lambda m: m.access_count, reverse=True)
        
        # Apply limit
        return results[:limit]
    
    def save(self) -> bool:
        """Save memory store to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_store_{timestamp}.pkl"
            filepath = os.path.join(self.storage_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.memories, f)
            
            # Update latest pointer
            latest_path = os.path.join(self.storage_dir, "latest.txt")
            with open(latest_path, 'w') as f:
                f.write(filename)
                
            return True
        except Exception as e:
            print(f"Error saving memory store: {e}")
            return False
    
    def load(self, filename: str = None) -> bool:
        """Load memory store from disk"""
        try:
            if filename is None:
                # Try to load latest
                latest_path = os.path.join(self.storage_dir, "latest.txt")
                if os.path.exists(latest_path):
                    with open(latest_path, 'r') as f:
                        filename = f.read().strip()
                else:
                    # No latest file, find most recent
                    files = [f for f in os.listdir(self.storage_dir) if f.startswith("memory_store_") and f.endswith(".pkl")]
                    if not files:
                        return False
                    files.sort(reverse=True)
                    filename = files[0]
            
            filepath = os.path.join(self.storage_dir, filename)
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'rb') as f:
                self.memories = pickle.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading memory store: {e}")
            return False

class EmotionEncoder:
    """Encodes emotional context for memories"""
    
    def __init__(self):
        # Emotional dimensions
        self.dimensions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "trust": 0.0,
            "anticipation": 0.0
        }
    
    def encode(self, content: Any, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Encode emotional tone for content"""
        # In a real implementation, this would use NLP or other analysis
        # For now, we'll use a simplified approach
        
        tone = self.dimensions.copy()
        
        # Extract emotional tone from context if provided
        if context and "emotional_state" in context:
            for dim, value in context["emotional_state"].items():
                if dim in tone:
                    tone[dim] = value
        
        # Analyze content for emotional markers
        if isinstance(content, str):
            # Very simple keyword-based analysis
            if "happy" in content.lower() or "joy" in content.lower():
                tone["joy"] += 0.3
            if "sad" in content.lower() or "unhappy" in content.lower():
                tone["sadness"] += 0.3
            if "angry" in content.lower() or "upset" in content.lower():
                tone["anger"] += 0.3
            if "afraid" in content.lower() or "scared" in content.lower():
                tone["fear"] += 0.3
            
            # Normalize values to 0-1 range
            for dim in tone:
                tone[dim] = max(0.0, min(1.0, tone[dim]))
        
        return tone

class MemoryService:
    """Memory service for managing system memories"""
    
    def __init__(self, storage_dir: str = None):
        self.store = MemoryStore(storage_dir)
        self.emotion_encoder = EmotionEncoder()
        self.running = False
        self._task = None
        self.heart_connected = False
        self.system_context = {}
    
    async def initialize(self):
        """Initialize the memory service"""
        # Load existing memories
        self.store.load()
        
        # Start background tasks
        self.running = True
        self._task = asyncio.create_task(self._maintenance_loop())
        
        return {"status": "initialized"}
    
    async def _maintenance_loop(self):
        """Background maintenance loop"""
        while self.running:
            # Periodically save memories
            self.store.save()
            
            # Wait before next maintenance cycle
            await asyncio.sleep(300)  # 5 minutes
    
    async def connect_to_heart(self, heart_service):
        """Connect to heart service for pulse synchronization"""
        try:
            # Register with heart service
            component_id = f"memory_{uuid.uuid4().hex[:8]}"
            await heart_service.register_component(component_id, self._on_heartbeat)
            self.heart_connected = True
            return {"status": "connected", "component_id": component_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _on_heartbeat(self, binary_packet):
        """Handle heartbeat from heart service"""
        # In a real implementation, we would parse the binary packet
        # For now, we'll just update our system context
        self.system_context["last_heartbeat"] = time.time()
    
    def create_memory(self, 
                     content: Any, 
                     memory_type: MemoryType = MemoryType.FACTUAL,
                     context_references: List[str] = None,
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new memory"""
        # Encode emotional tone
        emotional_tone = self.emotion_encoder.encode(content, self.system_context)
        
        # Create memory shard
        shard = MemoryShard(
            content=content,
            memory_type=memory_type,
            emotional_tone=emotional_tone,
            context_references=context_references,
            metadata=metadata
        )
        
        # Add to store
        shard_id = self.store.add(shard)
        
        return {"status": "created", "memory_id": shard_id}
    
    def retrieve_memory(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a memory by ID"""
        shard = self.store.get(memory_id)
        if shard:
            return {"status": "found", "memory": shard.to_dict()}
        else:
            return {"status": "not_found"}
    
    def search_memories(self,
                       memory_type: Optional[str] = None,
                       context_reference: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       limit: int = 10) -> Dict[str, Any]:
        """Search memories"""
        # Convert string memory type to enum if provided
        memory_type_enum = None
        if memory_type:
            try:
                memory_type_enum = MemoryType[memory_type.upper()]
            except KeyError:
                return {"status": "error", "message": f"Invalid memory type: {memory_type}"}
        
        # Query memories
        results = self.store.query(
            memory_type=memory_type_enum,
            context_reference=context_reference,
            metadata_filter=metadata,
            limit=limit
        )
        
        # Convert to dictionaries
        memory_dicts = [shard.to_dict() for shard in results]
        
        return {"status": "success", "memories": memory_dicts, "count": len(memory_dicts)}
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a memory"""
        shard = self.store.get(memory_id)
        if not shard:
            return {"status": "not_found"}
        
        # Apply updates
        if "content" in updates:
            shard.content = updates["content"]
            # Re-encode emotional tone if content changes
            shard.emotional_tone = self.emotion_encoder.encode(shard.content, self.system_context)
        
        if "context_references" in updates:
            shard.context_references = updates["context_references"]
        
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            shard.metadata.update(updates["metadata"])
        
        return {"status": "updated", "memory": shard.to_dict()}
    
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory"""
        success = self.store.remove(memory_id)
        if success:
            return {"status": "deleted"}
        else:
            return {"status": "not_found"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory service status"""
        return {
            "running": self.running,
            "heart_connected": self.heart_connected,
            "memory_count": len(self.store.memories),
            "last_save": self.system_context.get("last_save", None)
        }
    
    def stop(self) -> Dict[str, Any]:
        """Stop the memory service"""
        self.running = False
        if self._task:
            self._task.cancel()
        
        # Save memories before stopping
        self.store.save()
        
        return {"status": "stopped"}

# Singleton instance
_memory_service = None

def get_memory_service(storage_dir: str = None) -> MemoryService:
    """Get or create the memory service singleton"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(storage_dir)
    return _memory_service

async def initialize_memory_service(storage_dir: str = None) -> Dict[str, Any]:
    """Initialize the memory service"""
    service = get_memory_service(storage_dir)
    return await service.initialize()