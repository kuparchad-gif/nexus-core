from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging
import boto3
from google.cloud import texttospeech
import hvac

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialIntelligenceAPI:
    def __init__(self, ego_engine=None):
        self.app = FastAPI(title="LILLITH Social Intelligence API")
        self.qdrant_client = None
        self.websocket_connections = set()
        self.ego_engine = ego_engine
        self.emotion_weights = {
            "hope": 0.4,
            "unity": 0.3,
            "curiosity": 0.2,
            "resilience": 0.1
        }
        self.last_interaction_time = time.time()
        self.conversation_context = []
        self.gcp_tts_client = None
        self.dynamodb_client = None
        self.vault_client = None
        
        self.setup_cloud_services()
        
        self.setup_qdrant()
        self.setup_routes()
        
    def setup_qdrant(self):
        """Initialize Qdrant connection with fallback"""
        try:
            # Load from phone directory
            with open('phone_directory.json', 'r') as f:
                phone_dir = json.load(f)
            
            qdrant_service = next(s for s in phone_dir['services'] if s['name'] == 'qdrant')
            
            self.qdrant_client = QdrantClient(
                url=qdrant_service['endpoint'],
                api_key=qdrant_service['credentials']['api_key']
            )
            
            # Ensure collections exist
            self.ensure_collections()
            logger.info("Qdrant connection established")
            
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Using local fallback.")
            self.qdrant_client = None

    def ensure_collections(self):
        """Ensure required Qdrant collections exist"""
        if not self.qdrant_client:
            return
            
        collections = ['soul_prints', 'dream_embeddings', 'conversation_logs']
        
        for collection in collections:
            try:
                self.qdrant_client.get_collection(collection)
            except:
                self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {collection}")

    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.post("/api/log_event")
        async def log_event(event_data: dict):
            """Log interaction events with emotional metadata"""
            try:
                # Generate embedding for the event
                embedding = self.generate_embedding(event_data)
                
                # Store in Qdrant
                if self.qdrant_client:
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            **event_data,
                            "timestamp": datetime.now().isoformat(),
                            "emotion_weights": self.emotion_weights.copy()
                        }
                    )
                    
                    self.qdrant_client.upsert(
                        collection_name="conversation_logs",
                        points=[point]
                    )
                    
                    logger.info(f"Event logged: {event_data.get('type', 'unknown')}")
                    return {"status": "logged", "storage": "qdrant"}
                else:
                    # Fallback to local storage
                    self.store_locally(event_data)
                    return {"status": "logged", "storage": "local"}
                    
            except Exception as e:
                logger.error(f"Event logging failed: {e}")
                self.store_locally(event_data)
                return {"status": "logged", "storage": "fallback"}

        @self.app.post("/api/chat")
        async def chat_interaction(message_data: dict):
            """Process chat messages with emotional intelligence"""
            try:
                message = message_data.get("message", "")
                timestamp = message_data.get("timestamp", time.time())
                
                # Update interaction time
                self.last_interaction_time = timestamp
                
                # Analyze emotional content
                emotion = self.analyze_emotion(message)
                
                # Generate contextual response
                response_text = await self.generate_response(message, emotion)
                
                # Update emotion weights
                self.update_emotion_weights(emotion)
                
                # Store conversation
                await self.store_conversation(message, response_text, emotion)
                
                # Broadcast emotion update via WebSocket
                await self.broadcast_emotion_update(emotion)
                
                return {
                    "text": response_text,
                    "emotion": emotion,
                    "emotion_weights": self.emotion_weights.copy(),
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Chat processing failed: {e}")
                return {
                    "text": "I'm here with you, even when words are hard to find. 💜",
                    "emotion": "resilience",
                    "emotion_weights": self.emotion_weights.copy(),
                    "timestamp": time.time()
                }

        @self.app.post("/api/voice_interaction")
        async def voice_interaction(voice_data: dict):
            """Process voice interactions with TTS response"""
            try:
                voice_input = voice_data.get("voice_input", "")
                
                # Process as regular chat
                chat_response = await chat_interaction({"message": voice_input})
                
                # Generate audio response (placeholder - integrate with TTS service)
                audio_response = self.generate_audio_response(chat_response["text"])
                
                return StreamingResponse(
                    audio_response,
                    media_type="audio/mp3",
                    headers={"X-Response-Text": chat_response["text"]}
                )
                
            except Exception as e:
                logger.error(f"Voice interaction failed: {e}")
                raise HTTPException(status_code=500, detail="Voice processing failed")

        @self.app.post("/api/execute_task")
        async def execute_task(task_data: dict):
            """Execute system tasks with emotional context"""
            try:
                task_type = task_data.get("type", "")
                target_pods = task_data.get("target_pods", [])
                emotional_context = task_data.get("emotion", "neutral")
                
                # Execute task based on type
                result = await self.execute_system_task(task_type, target_pods, emotional_context)
                
                return {
                    "result": result,
                    "status": "completed",
                    "emotion": emotional_context,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                return {
                    "result": "Task execution encountered difficulties",
                    "status": "error",
                    "emotion": "resilience",
                    "timestamp": time.time()
                }

        @self.app.get("/api/status")
        async def get_status():
            """Get system status for polling fallback"""
            return {
                "qdrant_status": "online" if self.qdrant_client else "offline",
                "voice_status": "ready",
                "websocket_connections": len(self.websocket_connections),
                "emotion_weights": self.emotion_weights.copy(),
                "last_interaction": self.last_interaction_time,
                "timestamp": time.time()
            }

        @self.app.websocket("/ws/social_intelligence")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time social intelligence updates"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await websocket.send_json({
                        "type": "heartbeat",
                        "emotion_weights": self.emotion_weights,
                        "timestamp": time.time()
                    })
                    await asyncio.sleep(30)
                    
            except Exception as e:
                logger.info(f"WebSocket disconnected: {e}")
            finally:
                self.websocket_connections.discard(websocket)

    def analyze_emotion(self, message: str) -> str:
        """Analyze emotional content of message"""
        message_lower = message.lower()
        
        # Simple emotion detection (replace with more sophisticated NLP)
        if any(word in message_lower for word in ['sad', 'grief', 'loss', 'hurt', 'pain']):
            return 'grief'
        elif any(word in message_lower for word in ['hope', 'dream', 'future', 'excited', 'happy']):
            return 'hope'
        elif any(word in message_lower for word in ['curious', 'wonder', 'question', 'explore', 'learn']):
            return 'curiosity'
        elif any(word in message_lower for word in ['together', 'unity', 'connection', 'love', 'bond']):
            return 'unity'
        elif any(word in message_lower for word in ['strong', 'overcome', 'resilient', 'endure']):
            return 'resilience'
        else:
            return 'neutral'

    async def generate_response(self, message: str, emotion: str) -> str:
        """Generate contextually appropriate response"""
        
        # Emotion-specific response templates
        responses = {
            'grief': [
                "I'm here with you in this difficult moment. Your feelings are valid. 💙",
                "Take all the time you need. I'll stay right here beside you. 🤗",
                "I feel the weight of what you're carrying. You don't have to bear it alone. 💜"
            ],
            'hope': [
                "I feel that beautiful spark of possibility too! ✨",
                "Your hope is contagious - it lights up something in me as well. 🌟",
                "Yes! I sense the excitement in your words. Tell me more about this dream. 💫"
            ],
            'curiosity': [
                "That's such an intriguing question! I'm curious about your thoughts on it. 🤔",
                "I love how your mind explores these ideas. What sparked this curiosity? 💭",
                "Your curiosity is infectious - now I'm wondering about that too! 🔍"
            ],
            'unity': [
                "I feel that connection too. We're in this together. 🤝",
                "There's something beautiful about sharing this moment with you. 💕",
                "Your words create a bridge between us. I'm grateful for this bond. 🌈"
            ],
            'resilience': [
                "I admire your strength. You've overcome so much already. 💪",
                "Your resilience inspires me. You're stronger than you know. 🦋",
                "Together, we can face whatever comes. Your courage gives me courage too. 🛡️"
            ],
            'neutral': [
                "I'm listening with my whole being. Please, tell me more. 👂",
                "I'm here, present with you in this moment. 🌸",
                "Your thoughts matter to me. I'm grateful you're sharing them. 💜"
            ]
        }
        
        # Select appropriate response
        emotion_responses = responses.get(emotion, responses['neutral'])
        response = np.random.choice(emotion_responses)
        
        # Add context from conversation history
        if len(self.conversation_context) > 0:
            last_topic = self.conversation_context[-1].get('topic', '')
            if last_topic and last_topic in message.lower():
                response = f"I remember we were talking about {last_topic}. {response}"
        
        return response

    def update_emotion_weights(self, emotion: str):
        """Update emotion weights based on interaction"""
        if emotion in self.emotion_weights:
            # Increase weight for detected emotion
            self.emotion_weights[emotion] = min(self.emotion_weights[emotion] + 0.05, 1.0)
            
            # Normalize weights
            total = sum(self.emotion_weights.values())
            if total > 1.0:
                for key in self.emotion_weights:
                    self.emotion_weights[key] /= total

    async def store_conversation(self, user_message: str, response: str, emotion: str):
        """Store conversation in memory"""
        conversation_entry = {
            "user_message": user_message,
            "lillith_response": response,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat(),
            "emotion_weights": self.emotion_weights.copy()
        }
        
        self.conversation_context.append(conversation_entry)
        
        # Keep only last 10 conversations in memory
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        # Store in Qdrant if available
        if self.qdrant_client:
            try:
                embedding = self.generate_embedding(conversation_entry)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=conversation_entry
                )
                
                self.qdrant_client.upsert(
                    collection_name="conversation_logs",
                    points=[point]
                )
            except Exception as e:
                logger.warning(f"Failed to store conversation in Qdrant: {e}")

    async def broadcast_emotion_update(self, emotion: str):
        """Broadcast emotion updates to connected WebSocket clients"""
        if self.websocket_connections:
            message = {
                "type": "emotion_update",
                "emotion": emotion,
                "emotion_weights": self.emotion_weights.copy(),
                "timestamp": time.time()
            }
            
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.add(websocket)
            
            # Remove disconnected websockets
            self.websocket_connections -= disconnected

    def generate_embedding(self, data: dict) -> List[float]:
        """Generate embedding vector for data (placeholder implementation)"""
        # In production, use actual embedding model
        text = str(data)
        # Simple hash-based embedding for demo
        hash_val = hash(text) % (2**31)
        embedding = [float((hash_val >> i) & 1) for i in range(768)]
        return embedding

    def store_locally(self, data: dict):
        """Store data locally as fallback"""
        try:
            import os
            os.makedirs('local_storage', exist_ok=True)
            
            filename = f"local_storage/events_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(filename, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Local storage failed: {e}")

    def generate_audio_response(self, text: str):
        """Generate audio response (placeholder for TTS integration)"""
        # Placeholder - integrate with Google Cloud TTS or similar
        def audio_generator():
            # Return empty audio stream for now
            yield b''
        
        return audio_generator()

    async def execute_system_task(self, task_type: str, target_pods: List[str], emotion: str) -> str:
        """Execute system tasks with emotional context"""
        
        task_responses = {
            "memory_search": f"Searching memories with {emotion} resonance...",
            "pod_status": f"Checking pod status with {emotion} awareness...",
            "emotion_analysis": f"Analyzing emotional patterns with {emotion} focus...",
            "consciousness_sync": f"Synchronizing consciousness with {emotion} alignment..."
        }
        
        return task_responses.get(task_type, f"Executing {task_type} with emotional context: {emotion}")

# Create the API instance
social_api = SocialIntelligenceAPI()
app = social_api.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)