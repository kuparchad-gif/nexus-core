#!/usr/bin/env python
"""
Memory Service Launcher
- Starts the Memory service and its components
- Initializes Archiver for storage management
- Initializes Planner for memory classification
"""

import os
import sys
import asyncio
import json
import logging
import time
import zlib
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryLauncher")

# Constants
MEMORY_MAP_FILE = os.path.join(root_dir, "memory", "streams", "eden_memory_map.json")
HOT_STORAGE_PATH = os.path.join(root_dir, "memory", "hot")
COLD_STORAGE_PATH = os.path.join(root_dir, "memory", "cold")
ARCHIVE_PATH = os.path.join(root_dir, "memory", "archive")
EMOTIONAL_LOG_PATH = os.path.join(root_dir, "memory", "logs", "emotional_context.jsonl")

class ArchiveService:
    """Archive Service for memory storage and retrieval"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(root_dir, "Config", "archive_config.json")
        self.compression_enabled = True
        self.encryption_enabled = True
        self.running = False
        self.service_id = f"archive-{int(time.time())}"
        self.storage_locations = {
            "hot": HOT_STORAGE_PATH,
            "warm": os.path.join(ARCHIVE_PATH, "warm"),
            "cold": os.path.join(ARCHIVE_PATH, "cold")
        }
        
        # Create storage directories
        for location in self.storage_locations.values():
            os.makedirs(location, exist_ok=True)
        
        # Create logs directory
        os.makedirs(os.path.dirname(EMOTIONAL_LOG_PATH), exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        logger.info(f"[ArchiveService] Initialized with ID {self.service_id}")
    
    def _load_config(self):
        """Load service configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.compression_enabled = config.get("compression_enabled", True)
                self.encryption_enabled = config.get("encryption_enabled", True)
                
                logger.info(f"[ArchiveService] Loaded configuration")
                logger.info(f"[ArchiveService] Compression: {self.compression_enabled}")
                logger.info(f"[ArchiveService] Encryption: {self.encryption_enabled}")
            else:
                logger.info(f"[ArchiveService] Config not found at {self.config_path}, using defaults")
        except Exception as e:
            logger.error(f"[ArchiveService] Error loading config: {e}")
    
    def start(self):
        """Start the Archive Service"""
        if self.running:
            logger.info(f"[ArchiveService] Already running")
            return
        
        self.running = True
        logger.info(f"[ArchiveService] Started")
    
    def stop(self):
        """Stop the Archive Service"""
        if not self.running:
            logger.info(f"[ArchiveService] Already stopped")
            return
        
        self.running = False
        logger.info(f"[ArchiveService] Stopped")
    
    def store_memory(self, memory_key, memory_data, emotional_context=None, priority=1):
        """Store a memory with emotional context"""
        if not self.running:
            logger.info(f"[ArchiveService] Service not running")
            return False
        
        try:
            # Prepare metadata
            metadata = {
                "timestamp": time.time(),
                "key": memory_key,
                "priority": priority,
                "emotional_context": emotional_context or {},
                "pathway": emotional_context.get("pathway", "perception") if emotional_context else "perception"
            }
            
            # Convert memory data to bytes
            memory_bytes = json.dumps(memory_data).encode('utf-8')
            
            # Apply compression if enabled
            if self.compression_enabled:
                memory_bytes = zlib.compress(memory_bytes)
                metadata["compressed"] = True
            
            # Apply encryption if enabled
            if self.encryption_enabled:
                # Simple XOR encryption for demonstration
                key = b'EDEN_ARCHIVE_KEY'
                memory_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(memory_bytes)])
                metadata["encrypted"] = True
            
            # Determine storage location based on priority
            location = "hot" if priority > 1 else "warm"
            
            # Store in appropriate location
            storage_path = self.storage_locations[location]
            file_path = os.path.join(storage_path, f"{memory_key}.bin")
            meta_path = os.path.join(storage_path, f"{memory_key}.bin.meta")
            
            # Save memory data
            with open(file_path, 'wb') as f:
                f.write(memory_bytes)
            
            # Save metadata
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log emotional context if present
            if emotional_context:
                self._log_emotional_context(memory_key, emotional_context)
            
            logger.info(f"[ArchiveService] Stored memory {memory_key} in {location}")
            return True
        except Exception as e:
            logger.error(f"[ArchiveService] Error storing memory: {e}")
            return False
    
    def retrieve_memory(self, memory_key):
        """Retrieve a memory by key"""
        if not self.running:
            logger.info(f"[ArchiveService] Service not running")
            return None
        
        try:
            # Check each storage location
            for location, path in self.storage_locations.items():
                file_path = os.path.join(path, f"{memory_key}.bin")
                meta_path = os.path.join(path, f"{memory_key}.bin.meta")
                
                if os.path.exists(file_path) and os.path.exists(meta_path):
                    # Load metadata
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Load memory data
                    with open(file_path, 'rb') as f:
                        data_bytes = f.read()
                    
                    # Apply decryption if needed
                    if metadata.get("encrypted", False):
                        key = b'EDEN_ARCHIVE_KEY'
                        data_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(data_bytes)])
                    
                    # Apply decompression if needed
                    if metadata.get("compressed", False):
                        data_bytes = zlib.decompress(data_bytes)
                    
                    # Convert bytes to memory data
                    memory_data = json.loads(data_bytes.decode('utf-8'))
                    
                    # Add emotional context to result
                    result = {
                        "memory": memory_data,
                        "emotional_context": metadata.get("emotional_context", {}),
                        "timestamp": metadata.get("timestamp"),
                        "pathway": metadata.get("pathway", "perception")
                    }
                    
                    logger.info(f"[ArchiveService] Retrieved memory {memory_key} from {location}")
                    return result
            
            logger.info(f"[ArchiveService] Memory {memory_key} not found")
            return None
        except Exception as e:
            logger.error(f"[ArchiveService] Error retrieving memory: {e}")
            return None
    
    def _log_emotional_context(self, memory_key, emotional_context):
        """Log emotional context for analysis"""
        try:
            log_entry = {
                "timestamp": time.time(),
                "memory_key": memory_key,
                "emotional_context": emotional_context
            }
            
            with open(EMOTIONAL_LOG_PATH, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"[ArchiveService] Error logging emotional context: {e}")
    
    def get_storage_status(self):
        """Get status of all storage locations"""
        status = {}
        
        for location, path in self.storage_locations.items():
            if os.path.exists(path):
                # Count files
                files = [f for f in os.listdir(path) if f.endswith('.bin')]
                file_count = len(files)
                
                # Calculate size
                total_size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
                
                status[location] = {
                    "status": location,
                    "available": True,
                    "file_count": file_count,
                    "total_size": total_size,
                    "last_checked": time.time()
                }
            else:
                status[location] = {
                    "status": location,
                    "available": False,
                    "error": "path_not_found"
                }
        
        return status
    
    def optimize_storage(self):
        """Optimize storage by migrating data between hot, warm, and cold storage"""
        if not self.running:
            logger.info(f"[ArchiveService] Service not running")
            return {"success": False, "reason": "service_not_running"}
        
        try:
            # Get current time
            now = time.time()
            
            # Find memories to migrate
            migrations = {
                "hot_to_warm": [],
                "warm_to_cold": []
            }
            
            # Check hot storage for old memories
            hot_path = self.storage_locations["hot"]
            if os.path.exists(hot_path):
                for file in os.listdir(hot_path):
                    if file.endswith('.bin.meta'):
                        meta_path = os.path.join(hot_path, file)
                        memory_key = file[:-9]  # Remove .bin.meta
                        
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if memory is old (7 days)
                        if now - metadata["timestamp"] > 7 * 24 * 60 * 60:
                            migrations["hot_to_warm"].append(memory_key)
            
            # Check warm storage for old memories
            warm_path = self.storage_locations["warm"]
            if os.path.exists(warm_path):
                for file in os.listdir(warm_path):
                    if file.endswith('.bin.meta'):
                        meta_path = os.path.join(warm_path, file)
                        memory_key = file[:-9]  # Remove .bin.meta
                        
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if memory is old (30 days)
                        if now - metadata["timestamp"] > 30 * 24 * 60 * 60:
                            migrations["warm_to_cold"].append(memory_key)
            
            # Perform migrations
            for memory_key in migrations["hot_to_warm"]:
                self._migrate_memory(memory_key, "hot", "warm")
            
            for memory_key in migrations["warm_to_cold"]:
                self._migrate_memory(memory_key, "warm", "cold")
            
            logger.info(f"[ArchiveService] Storage optimization complete: {len(migrations['hot_to_warm'])} hot->warm, {len(migrations['warm_to_cold'])} warm->cold")
            
            return {
                "success": True,
                "migrations": {
                    "hot_to_warm": len(migrations["hot_to_warm"]),
                    "warm_to_cold": len(migrations["warm_to_cold"])
                }
            }
        except Exception as e:
            logger.error(f"[ArchiveService] Error optimizing storage: {e}")
            return {"success": False, "reason": str(e)}
    
    def _migrate_memory(self, memory_key, source, target):
        """Migrate a memory from one storage location to another"""
        source_path = self.storage_locations[source]
        target_path = self.storage_locations[target]
        
        source_file = os.path.join(source_path, f"{memory_key}.bin")
        source_meta = os.path.join(source_path, f"{memory_key}.bin.meta")
        
        target_file = os.path.join(target_path, f"{memory_key}.bin")
        target_meta = os.path.join(target_path, f"{memory_key}.bin.meta")
        
        # Check if source files exist
        if not os.path.exists(source_file) or not os.path.exists(source_meta):
            logger.error(f"[ArchiveService] Source files for {memory_key} not found")
            return False
        
        try:
            # Copy files to target
            with open(source_file, 'rb') as f:
                data = f.read()
            
            with open(target_file, 'wb') as f:
                f.write(data)
            
            with open(source_meta, 'r') as f:
                metadata = json.load(f)
            
            with open(target_meta, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Remove source files
            os.remove(source_file)
            os.remove(source_meta)
            
            logger.info(f"[ArchiveService] Migrated {memory_key} from {source} to {target}")
            return True
        except Exception as e:
            logger.error(f"[ArchiveService] Error migrating {memory_key}: {e}")
            return False

class PlannerService:
    """Planner component for memory classification"""
    
    def __init__(self, archive_service):
        self.archive_service = archive_service
        self.processing_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.processing_history = []
        self.running = False
    
    async def start(self):
        """Start the Planner Service"""
        logger.info("Starting Planner Service")
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._processing_worker())
        asyncio.create_task(self._result_worker())
        
        return True
    
    async def _processing_worker(self):
        """Worker that processes the queue and routes messages"""
        while self.running:
            try:
                # Get message from queue
                message = await self.processing_queue.get()
                
                # Determine if memory is emotional or logical
                memory_type, emotional_context = self._classify_memory(message)
                
                # Process according to type
                if memory_type == "emotional":
                    # Convert to binary and send to Services
                    binary_data = self._convert_to_binary(message, emotional_context)
                    
                    # Queue result for Services
                    await self.result_queue.put({
                        "type": "emotional",
                        "binary_data": binary_data,
                        "message": message,
                        "emotional_context": emotional_context
                    })
                else:
                    # Store logical memory
                    memory_key = f"memory-{int(time.time())}-{id(message)}"
                    self.archive_service.store_memory(
                        memory_key=memory_key,
                        memory_data=message,
                        emotional_context=emotional_context,
                        priority=1
                    )
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing worker: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _result_worker(self):
        """Worker that handles processed results"""
        while self.running:
            try:
                # Get processed result from queue
                result = await self.result_queue.get()
                
                if result["type"] == "emotional":
                    # In a real implementation, would send to Services
                    logger.info(f"Would send emotional memory to Services: {result['emotional_context']}")
                
                # Mark task as done
                self.result_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in result worker: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
    
    def _classify_memory(self, message):
        """Classify memory as emotional or logical"""
        # Simple classification based on content
        emotional_keywords = ["happy", "sad", "angry", "afraid", "love", "hate", "joy", "fear"]
        
        # Check if message contains emotional keywords
        if isinstance(message, dict) and "content" in message and isinstance(message["content"], str):
            content = message["content"].lower()
            if any(keyword in content for keyword in emotional_keywords):
                # Emotional memory
                emotional_context = self._analyze_emotional_tone(content)
                return "emotional", emotional_context
        
        # Default to logical
        return "logical", {}
    
    def _analyze_emotional_tone(self, content):
        """Analyze emotional tone of content"""
        # Simple analysis based on keywords
        tones = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "trust": 0.0,
            "disgust": 0.0,
            "anticipation": 0.0,
            "surprise": 0.0
        }
        
        # Check for keywords
        content_lower = content.lower()
        if "happy" in content_lower or "joy" in content_lower:
            tones["joy"] = 0.8
        if "sad" in content_lower or "unhappy" in content_lower:
            tones["sadness"] = 0.8
        if "angry" in content_lower or "mad" in content_lower:
            tones["anger"] = 0.8
        if "afraid" in content_lower or "scared" in content_lower:
            tones["fear"] = 0.8
        if "trust" in content_lower or "believe" in content_lower:
            tones["trust"] = 0.8
        if "disgust" in content_lower or "gross" in content_lower:
            tones["disgust"] = 0.8
        if "anticipate" in content_lower or "expect" in content_lower:
            tones["anticipation"] = 0.8
        if "surprise" in content_lower or "shock" in content_lower:
            tones["surprise"] = 0.8
        
        return {
            "tones": tones,
            "pathway": "experience",
            "intensity": max(tones.values()) if tones else 0.0
        }
    
    def _convert_to_binary(self, message, emotional_context):
        """Convert memory to binary format"""
        # In a real implementation, this would convert to a binary format
        # For now, just return a placeholder
        return b"BINARY_MEMORY_DATA"
    
    def process_memory(self, content, metadata=None):
        """Process a memory"""
        # Create message
        message = {
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Add to processing queue
        asyncio.create_task(self.processing_queue.put(message))
        
        return {"status": "queued", "timestamp": message["timestamp"]}
    
    def stop(self):
        """Stop the Planner Service"""
        logger.info("Stopping Planner Service")
        self.running = False

async def main():
    """Main entry point for Memory service"""
    logger.info("Starting Memory service...")
    
    try:
        # Create necessary directories
        os.makedirs(HOT_STORAGE_PATH, exist_ok=True)
        os.makedirs(COLD_STORAGE_PATH, exist_ok=True)
        os.makedirs(ARCHIVE_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(MEMORY_MAP_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(EMOTIONAL_LOG_PATH), exist_ok=True)
        
        # Initialize Archive Service
        archive_service = ArchiveService()
        archive_service.start()
        logger.info("Archive Service started")
        
        # Initialize Planner Service
        planner_service = PlannerService(archive_service)
        await planner_service.start()
        logger.info("Planner Service started")
        
        # Create test memory
        test_result = planner_service.process_memory(
            "This is a test memory to verify the system is working.",
            {"source": "system_init"}
        )
        logger.info(f"Test memory processed: {test_result}")
        
        # Keep the service running
        while True:
            # Periodically optimize storage
            await asyncio.sleep(3600)  # 1 hour
            archive_service.optimize_storage()
            
    except Exception as e:
        logger.error(f"Error in Memory service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())