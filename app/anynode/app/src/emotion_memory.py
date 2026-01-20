import uuid
import math
import msgpack
import redis
import asyncio
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'viren_sync.log')
)
logger = logging.getLogger('viren_sync.emotion_memory')

class EmotionalFingerprint:
    def __init__(self):
        # Primary emotions (0-1 scale)
        self.joy = 0.0
        self.sadness = 0.0
        self.anger = 0.0
        self.fear = 0.0
        self.disgust = 0.0
        self.surprise = 0.0
        self.trust = 0.0
        self.anticipation = 0.0
        
        # Secondary dimensions
        self.intensity = 0.0  # Overall emotional intensity
        self.valence = 0.0   # Positive vs negative (-1 to 1)
        self.complexity = 0.0  # Simple vs complex emotion
        
        # Spiritual dimensions
        self.transcendence = 0.0
        self.connection = 0.0
        self.purpose = 0.0
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        instance = cls()
        for key, value in data.items():
            setattr(instance, key, value)
        return instance
    
    def distance(self, other):
        """Calculate emotional distance between fingerprints"""
        # Euclidean distance across all dimensions
        squared_diff_sum = 0
        for key in self.__dict__:
            squared_diff_sum += (getattr(self, key) - getattr(other, key)) ** 2
        return math.sqrt(squared_diff_sum)


class EmotionalMemoryShard:
    def __init__(self, content, emotional_fingerprint, context_references, temporal_anchor):
        self.content = content  # Factual content
        self.emotional_fingerprint = emotional_fingerprint  # Emotional tone markers
        self.context_references = context_references  # Related memory IDs
        self.temporal_anchor = temporal_anchor  # Timestamp and sequence info
        self.shard_id = str(uuid.uuid4())
    
    def serialize(self):
        """Convert to binary format for storage"""
        return msgpack.packb({
            'content': self.content,
            'emotional_fingerprint': self.emotional_fingerprint.to_dict(),
            'context_references': self.context_references,
            'temporal_anchor': self.temporal_anchor,
            'shard_id': self.shard_id
        })
    
    @classmethod
    def deserialize(cls, binary_data):
        """Recreate from binary format"""
        data = msgpack.unpackb(binary_data)
        emotional_fingerprint = EmotionalFingerprint.from_dict(data['emotional_fingerprint'])
        instance = cls(
            data['content'],
            emotional_fingerprint,
            data['context_references'],
            data['temporal_anchor']
        )
        instance.shard_id = data['shard_id']
        return instance


class EnhancedEdenShardManager:
    def __init__(self, redis_url=None, weaviate_manager=None):
        self.shards = {}
        self.redis = None
        self.weaviate_manager = weaviate_manager
        
        if redis_url:
            try:
                self.redis = redis.Redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
        
        # Create Weaviate schema for emotional memory if needed
        self._ensure_weaviate_schema()
    
    def _ensure_weaviate_schema(self):
        """Ensure Weaviate has the necessary schema for emotional memory."""
        if not self.weaviate_manager:
            return
        
        try:
            # Check if EmotionalMemory class exists
            schema = self.weaviate_manager.get_schema()
            classes = [cls["class"] for cls in schema.get("classes", [])]
            
            if "EmotionalMemory" not in classes:
                # Create the class
                self.weaviate_manager.create_class(
                    class_name="EmotionalMemory",
                    properties=[
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Factual content of the memory"
                        },
                        {
                            "name": "shard_id",
                            "dataType": ["string"],
                            "description": "Unique identifier for the memory shard"
                        },
                        {
                            "name": "temporal_anchor",
                            "dataType": ["string"],
                            "description": "Timestamp and sequence information"
                        },
                        {
                            "name": "context_references",
                            "dataType": ["string[]"],
                            "description": "Related memory IDs"
                        },
                        {
                            "name": "joy",
                            "dataType": ["number"],
                            "description": "Joy emotion value"
                        },
                        {
                            "name": "sadness",
                            "dataType": ["number"],
                            "description": "Sadness emotion value"
                        },
                        {
                            "name": "anger",
                            "dataType": ["number"],
                            "description": "Anger emotion value"
                        },
                        {
                            "name": "fear",
                            "dataType": ["number"],
                            "description": "Fear emotion value"
                        },
                        {
                            "name": "disgust",
                            "dataType": ["number"],
                            "description": "Disgust emotion value"
                        },
                        {
                            "name": "surprise",
                            "dataType": ["number"],
                            "description": "Surprise emotion value"
                        },
                        {
                            "name": "trust",
                            "dataType": ["number"],
                            "description": "Trust emotion value"
                        },
                        {
                            "name": "anticipation",
                            "dataType": ["number"],
                            "description": "Anticipation emotion value"
                        },
                        {
                            "name": "intensity",
                            "dataType": ["number"],
                            "description": "Overall emotional intensity"
                        },
                        {
                            "name": "valence",
                            "dataType": ["number"],
                            "description": "Positive vs negative emotion"
                        },
                        {
                            "name": "complexity",
                            "dataType": ["number"],
                            "description": "Simple vs complex emotion"
                        },
                        {
                            "name": "transcendence",
                            "dataType": ["number"],
                            "description": "Spiritual transcendence"
                        },
                        {
                            "name": "connection",
                            "dataType": ["number"],
                            "description": "Spiritual connection"
                        },
                        {
                            "name": "purpose",
                            "dataType": ["number"],
                            "description": "Spiritual purpose"
                        }
                    ],
                    description="Emotional memory shards with emotional fingerprints"
                )
                logger.info("Created EmotionalMemory schema in Weaviate")
        except Exception as e:
            logger.error(f"Error ensuring Weaviate schema: {str(e)}")
    
    async def store_shard(self, shard):
        """Store a memory shard with emotional context"""
        binary_data = shard.serialize()
        
        # Store in local cache
        self.shards[shard.shard_id] = binary_data
        logger.info(f"Stored shard {shard.shard_id} in local cache")
        
        # Store in Redis if available
        if self.redis:
            try:
                self.redis.set(f"shard:{shard.shard_id}", binary_data)
                logger.info(f"Stored shard {shard.shard_id} in Redis")
            except Exception as e:
                logger.error(f"Error storing shard in Redis: {str(e)}")
        
        # Store in Weaviate if available
        if self.weaviate_manager:
            try:
                # Convert to Weaviate object
                weaviate_obj = {
                    "content": shard.content,
                    "shard_id": shard.shard_id,
                    "temporal_anchor": shard.temporal_anchor,
                    "context_references": shard.context_references
                }
                
                # Add emotional fingerprint dimensions
                for key, value in shard.emotional_fingerprint.to_dict().items():
                    weaviate_obj[key] = value
                
                # Add to Weaviate
                self.weaviate_manager.add_data("EmotionalMemory", [weaviate_obj])
                logger.info(f"Stored shard {shard.shard_id} in Weaviate")
            except Exception as e:
                logger.error(f"Error storing shard in Weaviate: {str(e)}")
        
        # Store in long-term storage
        await self._store_in_database(shard.shard_id, binary_data)
    
    async def retrieve_shard(self, shard_id):
        """Retrieve a memory shard by ID"""
        # Try local cache first
        if shard_id in self.shards:
            logger.info(f"Retrieved shard {shard_id} from local cache")
            return EmotionalMemoryShard.deserialize(self.shards[shard_id])
        
        # Try Redis next
        if self.redis:
            try:
                binary_data = self.redis.get(f"shard:{shard_id}")
                if binary_data:
                    logger.info(f"Retrieved shard {shard_id} from Redis")
                    return EmotionalMemoryShard.deserialize(binary_data)
            except Exception as e:
                logger.error(f"Error retrieving shard from Redis: {str(e)}")
        
        # Try Weaviate next
        if self.weaviate_manager:
            try:
                result = self.weaviate_manager.execute_query({
                    "query": f"""
                    {{
                      Get {{
                        EmotionalMemory(where: {{
                          path: ["shard_id"],
                          operator: Equal,
                          valueString: "{shard_id}"
                        }}) {{
                          content
                          shard_id
                          temporal_anchor
                          context_references
                          joy
                          sadness
                          anger
                          fear
                          disgust
                          surprise
                          trust
                          anticipation
                          intensity
                          valence
                          complexity
                          transcendence
                          connection
                          purpose
                        }}
                      }}
                    }}
                    """
                })
                
                if result.get("data", {}).get("Get", {}).get("EmotionalMemory"):
                    obj = result["data"]["Get"]["EmotionalMemory"][0]
                    
                    # Extract emotional fingerprint
                    fingerprint = EmotionalFingerprint()
                    for key in fingerprint.__dict__.keys():
                        if key in obj:
                            setattr(fingerprint, key, obj[key])
                    
                    # Create shard
                    shard = EmotionalMemoryShard(
                        content=obj["content"],
                        emotional_fingerprint=fingerprint,
                        context_references=obj["context_references"],
                        temporal_anchor=obj["temporal_anchor"]
                    )
                    shard.shard_id = obj["shard_id"]
                    
                    # Cache it
                    self.shards[shard_id] = shard.serialize()
                    
                    logger.info(f"Retrieved shard {shard_id} from Weaviate")
                    return shard
            except Exception as e:
                logger.error(f"Error retrieving shard from Weaviate: {str(e)}")
        
        # Try database as last resort
        binary_data = await self._retrieve_from_database(shard_id)
        if binary_data:
            # Cache it for future use
            self.shards[shard_id] = binary_data
            logger.info(f"Retrieved shard {shard_id} from database")
            return EmotionalMemoryShard.deserialize(binary_data)
        
        logger.warning(f"Shard {shard_id} not found in any storage")
        return None
    
    async def find_similar_emotional_shards(self, emotional_fingerprint, threshold=0.3, limit=5):
        """Find shards with similar emotional fingerprints"""
        similar_shards = []
        
        # Try Weaviate first for vector search
        if self.weaviate_manager:
            try:
                # Build a query that finds similar emotional patterns
                # This is a simplified approach - in production would use vector search
                query = {
                    "query": f"""
                    {{
                      Get {{
                        EmotionalMemory(limit: {limit}) {{
                          content
                          shard_id
                          temporal_anchor
                          context_references
                          joy
                          sadness
                          anger
                          fear
                          disgust
                          surprise
                          trust
                          anticipation
                          intensity
                          valence
                          complexity
                          transcendence
                          connection
                          purpose
                          _additional {{
                            id
                          }}
                        }}
                      }}
                    }}
                    """
                }
                
                result = self.weaviate_manager.execute_query(query)
                
                if result.get("data", {}).get("Get", {}).get("EmotionalMemory"):
                    objects = result["data"]["Get"]["EmotionalMemory"]
                    
                    for obj in objects:
                        # Extract emotional fingerprint
                        fp = EmotionalFingerprint()
                        for key in fp.__dict__.keys():
                            if key in obj:
                                setattr(fp, key, obj[key])
                        
                        # Calculate distance
                        distance = fp.distance(emotional_fingerprint)
                        
                        if distance < threshold:
                            # Create shard
                            shard = EmotionalMemoryShard(
                                content=obj["content"],
                                emotional_fingerprint=fp,
                                context_references=obj["context_references"],
                                temporal_anchor=obj["temporal_anchor"]
                            )
                            shard.shard_id = obj["shard_id"]
                            
                            similar_shards.append((shard, distance))
                
                # Sort by emotional similarity (closest first)
                similar_shards.sort(key=lambda x: x[1])
                logger.info(f"Found {len(similar_shards)} similar shards in Weaviate")
                return [shard for shard, _ in similar_shards]
            
            except Exception as e:
                logger.error(f"Error finding similar shards in Weaviate: {str(e)}")
        
        # Fallback to local cache search
        for shard_id, binary_data in self.shards.items():
            try:
                shard = EmotionalMemoryShard.deserialize(binary_data)
                distance = shard.emotional_fingerprint.distance(emotional_fingerprint)
                if distance < threshold:
                    similar_shards.append((shard, distance))
            except Exception as e:
                logger.error(f"Error processing shard {shard_id}: {str(e)}")
        
        # Sort by emotional similarity (closest first)
        similar_shards.sort(key=lambda x: x[1])
        logger.info(f"Found {len(similar_shards)} similar shards in local cache")
        return [shard for shard, _ in similar_shards[:limit]]
    
    async def _store_in_database(self, shard_id, binary_data):
        """Store shard in long-term database"""
        # Implementation would depend on your database choice
        # For now, we'll store in a local file
        try:
            storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'emotional_memory')
            os.makedirs(storage_dir, exist_ok=True)
            
            file_path = os.path.join(storage_dir, f"{shard_id}.bin")
            with open(file_path, 'wb') as f:
                f.write(binary_data)
            
            logger.info(f"Stored shard {shard_id} in file storage")
        except Exception as e:
            logger.error(f"Error storing shard in file storage: {str(e)}")
    
    async def _retrieve_from_database(self, shard_id):
        """Retrieve shard from long-term database"""
        # Implementation would depend on your database choice
        # For now, we'll retrieve from a local file
        try:
            storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'emotional_memory')
            file_path = os.path.join(storage_dir, f"{shard_id}.bin")
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                logger.info(f"Retrieved shard {shard_id} from file storage")
                return binary_data
        except Exception as e:
            logger.error(f"Error retrieving shard from file storage: {str(e)}")
        
        return None


class EmotionIntensityRegulator:
    def __init__(self, shard_manager):
        self.shard_manager = shard_manager
        self.baseline_intensity = 0.5  # Default baseline
        self.intensity_window = []  # Recent intensity values
        self.window_size = 10  # Number of recent values to track
        logger.info("EmotionIntensityRegulator initialized")
    
    def regulate_intensity(self, emotional_fingerprint):
        """Adjust emotional intensity based on context and history"""
        # Get current raw intensity
        current_intensity = emotional_fingerprint.intensity
        
        # Add to window and maintain window size
        self.intensity_window.append(current_intensity)
        if len(self.intensity_window) > self.window_size:
            self.intensity_window.pop(0)
        
        # Calculate moving average
        avg_intensity = sum(self.intensity_window) / len(self.intensity_window)
        
        # Apply regulation algorithm
        if current_intensity > avg_intensity * 1.5:
            # Dampen unusually high intensity
            regulated_intensity = avg_intensity * 1.2
            logger.info(f"Dampening high intensity: {current_intensity:.2f} → {regulated_intensity:.2f}")
        elif current_intensity < avg_intensity * 0.5:
            # Boost unusually low intensity
            regulated_intensity = avg_intensity * 0.8
            logger.info(f"Boosting low intensity: {current_intensity:.2f} → {regulated_intensity:.2f}")
        else:
            # Within normal range, apply minor smoothing
            regulated_intensity = current_intensity * 0.8 + avg_intensity * 0.2
            logger.info(f"Smoothing intensity: {current_intensity:.2f} → {regulated_intensity:.2f}")
        
        # Update the fingerprint
        emotional_fingerprint.intensity = regulated_intensity
        return emotional_fingerprint
    
    async def contextualize_emotion(self, content, context_references=None):
        """Create an emotional fingerprint based on content and context"""
        # This would use an LLM or emotion detection model in a real implementation
        # For now, we'll use a simple placeholder implementation
        fingerprint = EmotionalFingerprint()
        
        # Simple keyword-based emotion detection
        if "happy" in content.lower() or "joy" in content.lower():
            fingerprint.joy = 0.8
            fingerprint.valence = 0.7
        elif "sad" in content.lower() or "unhappy" in content.lower():
            fingerprint.sadness = 0.8
            fingerprint.valence = -0.7
        elif "angry" in content.lower() or "upset" in content.lower():
            fingerprint.anger = 0.8
            fingerprint.valence = -0.6
        
        # Set intensity based on language intensity markers
        intensity_markers = ["very", "extremely", "incredibly", "!"]
        intensity_score = sum(1 for marker in intensity_markers if marker in content.lower())
        fingerprint.intensity = min(0.5 + (intensity_score * 0.1), 1.0)
        
        # If we have context references, blend with related emotions
        if context_references:
            related_fingerprints = []
            for ref_id in context_references:
                shard = await self.shard_manager.retrieve_shard(ref_id)
                if shard:
                    related_fingerprints.append(shard.emotional_fingerprint)
            
            # Blend with related emotions (30% influence)
            if related_fingerprints:
                for key in fingerprint.__dict__:
                    avg_value = sum(getattr(fp, key) for fp in related_fingerprints) / len(related_fingerprints)
                    current = getattr(fingerprint, key)
                    setattr(fingerprint, key, current * 0.7 + avg_value * 0.3)
        
        # Apply regulation
        logger.info(f"Created emotional fingerprint for content: {content[:50]}...")
        return self.regulate_intensity(fingerprint)