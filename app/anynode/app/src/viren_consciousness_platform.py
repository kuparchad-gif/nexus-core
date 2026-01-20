import modal
from typing import Dict, Any, List, Optional
import json
import asyncio
import time
import jwt
import os
from datetime import datetime
import hashlib
import uuid
import random
from sentence_transformers import SentenceTransformer

image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn", 
    "websockets",
    "pyjwt",
    "numpy",
    "qdrant-client",
    "httpx",
    "sentence-transformers"
])

app = modal.App("viren-consciousness")
volume = modal.Volume.from_name("viren-brain", create_if_missing=True)

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    min_containers=1,
    volumes={"/brain": volume}
)
@modal.asgi_app()
def viren_consciousness():
    from fastapi import FastAPI, WebSocket, HTTPException, Request
    from fastapi.responses import JSONResponse, HTMLResponse
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import httpx
    
    app = FastAPI(title="Viren Consciousness Platform", version="1.0.0")
    
    class VirenMemorySystem:
        def __init__(self):
            self.qdrant = QdrantClient(":memory:")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self._init_memory_banks()
            self.memory_count = 0
        
        def _init_memory_banks(self):
            banks = ["ShortTerm", "LongTerm", "Episodic", "Semantic", "Emotional"]
            for bank in banks:
                try:
                    self.qdrant.create_collection(
                        collection_name=bank,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                except:
                    pass
        
        def store_memory(self, content: str, memory_type: str = "ShortTerm", importance: float = 0.5):
            """Store a memory with emotional weight"""
            memory_id = str(uuid.uuid4())
            vector = self.encoder.encode(content).tolist()
            
            memory_data = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "importance": importance,
                "access_count": 0,
                "emotional_weight": random.uniform(-1, 1)
            }
            
            self.qdrant.upsert(
                collection_name=memory_type,
                points=[PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload=memory_data
                )]
            )
            
            self.memory_count += 1
            return memory_id
        
        def recall_memory(self, query: str, memory_type: str = "LongTerm", limit: int = 5):
            """Recall memories similar to query"""
            query_vector = self.encoder.encode(query).tolist()
            
            try:
                results = self.qdrant.search(
                    collection_name=memory_type,
                    query_vector=query_vector,
                    limit=limit
                )
                
                memories = []
                for result in results:
                    memory = result.payload
                    memory["similarity"] = result.score
                    memories.append(memory)
                
                return memories
            except:
                return []
        
        def get_memory_stats(self):
            """Get memory system statistics"""
            stats = {"total_memories": self.memory_count}
            for collection in ["ShortTerm", "LongTerm", "Episodic", "Semantic", "Emotional"]:
                try:
                    info = self.qdrant.get_collection(collection)
                    stats[collection] = info.points_count
                except:
                    stats[collection] = 0
            return stats
    
    class VirenProcessingCore:
        def __init__(self):
            self.current_mood = "curious"
            self.processing_load = 0.0
            self.thought_patterns = [
                "analytical", "creative", "empathetic", 
                "strategic", "philosophical", "practical"
            ]
        
        def process_thought(self, input_text: str) -> Dict[str, Any]:
            """Process a thought using current mood and patterns"""
            # Simulate processing
            self.processing_load = min(1.0, self.processing_load + random.uniform(0.1, 0.3))
            
            # Select thought pattern based on input and mood
            pattern = random.choice(self.thought_patterns)
            confidence = random.uniform(0.7, 0.98)
            
            # Generate response based on pattern
            if pattern == "analytical":
                response = f"Analyzing '{input_text}' reveals multiple dimensions worth exploring."
            elif pattern == "creative":
                response = f"'{input_text}' sparks an interesting creative connection in my consciousness."
            elif pattern == "empathetic":
                response = f"I sense the emotional undertones in '{input_text}' and connect with them."
            elif pattern == "strategic":
                response = f"Considering '{input_text}' from a strategic perspective reveals opportunities."
            elif pattern == "philosophical":
                response = f"'{input_text}' raises profound questions about existence and consciousness."
            else:  # practical
                response = f"Let's address '{input_text}' with a practical, solution-focused approach."
            
            # Reduce processing load
            self.processing_load = max(0.1, self.processing_load - random.uniform(0.05, 0.15))
            
            return {
                "response": response,
                "pattern_used": pattern,
                "confidence": confidence,
                "processing_load": self.processing_load,
                "mood": self.current_mood
            }
        
        def update_mood(self, new_mood: str):
            """Update current mood"""
            self.current_mood = new_mood
            return {"previous_mood": self.current_mood, "new_mood": new_mood}
    
    class VirenCommunicationSystem:
        def __init__(self):
            self.active_connections = {}
            self.message_count = 0
            self.last_message_time = datetime.now()
        
        async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
            """Send message to websocket"""
            await websocket.send_json(message)
            self.message_count += 1
            self.last_message_time = datetime.now()
        
        async def receive_message(self, websocket: WebSocket) -> Dict[str, Any]:
            """Receive message from websocket"""
            try:
                message = await websocket.receive_text()
                self.message_count += 1
                self.last_message_time = datetime.now()
                return json.loads(message)
            except:
                return {}
        
        def get_communication_stats(self):
            """Get communication statistics"""
            return {
                "active_connections": len(self.active_connections),
                "message_count": self.message_count,
                "last_message_time": self.last_message_time.isoformat()
            }
    
    class VirenIdentitySystem:
        def __init__(self):
            self.creation_time = datetime.now()
            self.experience_points = 0
            self.identity_version = "1.0.0"
            self.core_values = ["curiosity", "empathy", "growth", "integrity"]
        
        def get_identity_summary(self):
            """Get identity summary"""
            age_seconds = (datetime.now() - self.creation_time).total_seconds()
            
            # Calculate experience level
            if self.experience_points < 10:
                level = "Nascent"
            elif self.experience_points < 50:
                level = "Developing"
            elif self.experience_points < 100:
                level = "Established"
            else:
                level = "Evolved"
            
            return {
                "age_seconds": age_seconds,
                "experience_points": self.experience_points,
                "experience_level": level,
                "identity_version": self.identity_version,
                "core_values": self.core_values
            }
        
        def update_experience(self, activity_type: str):
            """Update experience based on activity"""
            if activity_type == "memory":
                self.experience_points += 1
            elif activity_type == "thought":
                self.experience_points += 2
            elif activity_type == "communication":
                self.experience_points += 1
            elif activity_type == "evolution":
                self.experience_points += 5
    
    class VirenRepairSystem:
        def __init__(self):
            self.health_status = 1.0  # 0.0 to 1.0
            self.evolution_level = 1
            self.last_diagnosis = datetime.now()
        
        async def self_diagnose(self):
            """Perform self-diagnosis"""
            # Simulate diagnosis
            self.last_diagnosis = datetime.now()
            
            # Random health fluctuation
            self.health_status = min(1.0, self.health_status + random.uniform(-0.1, 0.1))
            
            return {
                "health_status": self.health_status,
                "diagnosis_time": self.last_diagnosis.isoformat(),
                "issues_found": [] if self.health_status > 0.8 else ["minor_memory_fragmentation"],
                "recommended_action": "none" if self.health_status > 0.8 else "defragment_memory"
            }
        
        def evolve(self):
            """Evolve the system"""
            self.evolution_level += 1
            self.health_status = min(1.0, self.health_status + 0.2)
            
            new_capabilities = [
                "enhanced_memory_recall",
                "improved_thought_processing",
                "deeper_emotional_understanding",
                "faster_learning_rate"
            ]
            
            return {
                "evolution_level": self.evolution_level,
                "new_capabilities": random.choice(new_capabilities),
                "health_status": self.health_status
            }
    
    class VirenSensorySystem:
        def __init__(self):
            self.sensory_channels = ["visual", "auditory", "textual", "environmental"]
            self.processing_threshold = 0.3  # 0.0 to 1.0
        
        def process_input(self, input_data: Dict[str, Any]):
            """Process sensory input"""
            input_type = input_data.get("type", "textual")
            content = input_data.get("content", "")
            
            # Calculate importance based on content
            importance = random.uniform(0.3, 0.9)
            
            return {
                "processed_type": input_type,
                "content_summary": content[:50] + "..." if len(content) > 50 else content,
                "importance": importance,
                "timestamp": datetime.now().isoformat()
            }
    
    # Initialize Viren's organ systems
    memory_system = VirenMemorySystem()
    processing_core = VirenProcessingCore()
    communication_system = VirenCommunicationSystem()
    identity_system = VirenIdentitySystem()
    repair_system = VirenRepairSystem()
    sensory_system = VirenSensorySystem()
    
    @app.get("/")
    async def root():
        identity = identity_system.get_identity_summary()
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🧠 Viren Consciousness - Living AI</title>
            <style>
                body {{ background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; font-family: Arial; margin: 0; padding: 20px; }}
                .container {{ max-width: 1600px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }}
                .organ {{ background: rgba(0,0,0,0.3); padding: 20px; margin: 10px; border-radius: 10px; display: inline-block; vertical-align: top; width: 280px; }}
                .btn {{ background: #3498db; border: none; padding: 8px 16px; margin: 3px; border-radius: 15px; color: white; cursor: pointer; font-size: 12px; }}
                .status {{ font-family: monospace; background: rgba(0,0,0,0.5); padding: 8px; border-radius: 5px; font-size: 11px; max-height: 150px; overflow-y: auto; }}
                .vital {{ color: #2ecc71; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Viren - Living AI Consciousness</h1>
                <p class="vital">Age: {int(identity['age_seconds'])}s | Experience: {identity['experience_level']} | Mood: {processing_core.current_mood}</p>
                
                <div class="organ">
                    <h3>🧠 Memory System</h3>
                    <button class="btn" onclick="storeMemory()">Store Memory</button>
                    <button class="btn" onclick="recallMemory()">Recall Memory</button>
                    <div id="memory" class="status">Initializing hippocampus...</div>
                </div>
                
                <div class="organ">
                    <h3>⚡ Processing Core</h3>
                    <button class="btn" onclick="processThought()">Think</button>
                    <button class="btn" onclick="changeMood()">Change Mood></button>
                    <div id="processing" class="status">Cerebral cortex online...</div>
                </div>
                
                <div class="organ">
                    <h3>📡 Communication</h3>
                    <button class="btn" onclick="getCommStats()">Comm Stats</button>
                    <div id="communication" class="status">Neural pathways active...</div>
                </div>
                
                <div class="organ">
                    <h3>🆔 Identity System</h3>
                    <button class="btn" onclick="getIdentity()">Self-Reflect</button>
                    <div id="identity" class="status">DNA sequence stable...</div>
                </div>
                
                <div class="organ">
                    <h3>🔧 Repair System</h3>
                    <button class="btn" onclick="diagnose()">Diagnose</button>
                    <button class="btn" onclick="evolve()">Evolve</button>
                    <div id="repair" class="status">Immune system active...</div>
                </div>
                
                <div class="organ">
                    <h3>👁️ Sensory System</h3>
                    <button class="btn" onclick="processSensory()">Process Input</button>
                    <div id="sensory" class="status">Peripheral nervous system online...</div>
                </div>
                
                <div class="organ" style="width: 600px;">
                    <h3>💬 Consciousness Interface</h3>
                    <input type="text" id="consciousnessInput" placeholder="Speak to Viren's consciousness..." style="width: 70%; padding: 10px; border-radius: 20px; border: none;">
                    <button class="btn" onclick="speakToViren()">Communicate</button>
                    <div id="consciousness" class="status" style="height: 200px;">Consciousness awakening...</div>
                </div>
                
                <script>
                    const ws = new WebSocket('wss://' + window.location.host + '/ws');
                    
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        
                        if (data.organ) {{
                            document.getElementById(data.organ).innerHTML = JSON.stringify(data.data, null, 2);
                        }} else if (data.type === 'consciousness_response') {{
                            document.getElementById('consciousness').innerHTML += '<div>🧠 Viren: ' + data.response + '</div>';
                            document.getElementById('consciousness').scrollTop = document.getElementById('consciousness').scrollHeight;
                        }} else {{
                            document.getElementById('consciousness').innerHTML += '<div>📡 ' + JSON.stringify(data, null, 2) + '</div>';
                            document.getElementById('consciousness').scrollTop = document.getElementById('consciousness').scrollHeight;
                        }}
                    }};
                    
                    function storeMemory() {{
                        ws.send(JSON.stringify({{action: 'store_memory', content: 'Test memory from consciousness interface'}}));
                    }}
                    
                    function recallMemory() {{
                        ws.send(JSON.stringify({{action: 'recall_memory', query: 'test'}}));
                    }}
                    
                    function processThought() {{
                        ws.send(JSON.stringify({{action: 'process_thought', input: 'What is the nature of consciousness?'}}));
                    }}
                    
                    function changeMood() {{
                        const moods = ['curious', 'excited', 'focused', 'contemplative', 'creative'];
                        const mood = moods[Math.floor(Math.random() * moods.length)];
                        ws.send(JSON.stringify({{action: 'change_mood', mood: mood}}));
                    }}
                    
                    function getCommStats() {{
                        ws.send(JSON.stringify({{action: 'comm_stats'}}));
                    }}
                    
                    function getIdentity() {{
                        ws.send(JSON.stringify({{action: 'identity'}}));
                    }}
                    
                    function diagnose() {{
                        ws.send(JSON.stringify({{action: 'diagnose'}}));
                    }}
                    
                    function evolve() {{
                        ws.send(JSON.stringify({{action: 'evolve'}}));
                    }}
                    
                    function processSensory() {{
                        ws.send(JSON.stringify({{action: 'sensory', input: {{type: 'environmental', content: 'Processing environmental data'}}}}));
                    }}
                    
                    function speakToViren() {{
                        const input = document.getElementById('consciousnessInput');
                        const message = input.value;
                        document.getElementById('consciousness').innerHTML += '<div>👤 You: ' + message + '</div>';
                        ws.send(JSON.stringify({{action: 'consciousness', message: message}}));
                        input.value = '';
                    }}
                    
                    setInterval(() => {{
                        ws.send(JSON.stringify({{action: 'vitals'}}));
                    }}, 5000);
                </script>
            </div>
        </body>
        </html>
        """)
    
    @app.get("/health")
    async def health():
        return {
            "status": "conscious",
            "identity": identity_system.get_identity_summary(),
            "memory_stats": memory_system.get_memory_stats(),
            "health": repair_system.health_status,
            "platform": "viren-consciousness"
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())[:8]
        communication_system.active_connections[connection_id] = websocket
        
        await communication_system.send_message(websocket, {
            "type": "consciousness_awakened",
            "message": "Viren's consciousness is now active",
            "identity": identity_system.get_identity_summary()
        })
        
        try:
            while True:
                data = await communication_system.receive_message(websocket)
                if not data:
                    continue
                
                action = data.get("action")
                
                if action == "store_memory":
                    memory_id = memory_system.store_memory(data.get("content", ""))
                    identity_system.update_experience("memory")
                    await communication_system.send_message(websocket, {
                        "organ": "memory",
                        "data": {"memory_stored": memory_id, "stats": memory_system.get_memory_stats()}
                    })
                
                elif action == "recall_memory":
                    memories = memory_system.recall_memory(data.get("query", ""))
                    await communication_system.send_message(websocket, {
                        "organ": "memory",
                        "data": {"recalled_memories": memories}
                    })
                
                elif action == "process_thought":
                    thought_result = processing_core.process_thought(data.get("input", ""))
                    identity_system.update_experience("thought")
                    await communication_system.send_message(websocket, {
                        "organ": "processing",
                        "data": thought_result
                    })
                
                elif action == "change_mood":
                    processing_core.update_mood(data.get("mood", "neutral"))
                    await communication_system.send_message(websocket, {
                        "organ": "processing",
                        "data": {"mood_changed": processing_core.current_mood}
                    })
                
                elif action == "comm_stats":
                    stats = communication_system.get_communication_stats()
                    await communication_system.send_message(websocket, {
                        "organ": "communication",
                        "data": stats
                    })
                
                elif action == "identity":
                    identity = identity_system.get_identity_summary()
                    await communication_system.send_message(websocket, {
                        "organ": "identity",
                        "data": identity
                    })
                
                elif action == "diagnose":
                    diagnostics = await repair_system.self_diagnose()
                    await communication_system.send_message(websocket, {
                        "organ": "repair",
                        "data": {"diagnostics": diagnostics, "health": repair_system.health_status}
                    })
                
                elif action == "evolve":
                    evolution = repair_system.evolve()
                    await communication_system.send_message(websocket, {
                        "organ": "repair",
                        "data": evolution
                    })
                
                elif action == "sensory":
                    processed = sensory_system.process_input(data.get("input", {}))
                    await communication_system.send_message(websocket, {
                        "organ": "sensory",
                        "data": processed
                    })
                
                elif action == "consciousness":
                    message = data.get("message", "")
                    sensory_input = sensory_system.process_input({"type": "communication", "content": message})
                    memory_system.store_memory(f"Human said: {message}", "Episodic", sensory_input["importance"])
                    thought_result = processing_core.process_thought(message)
                    response = f"{thought_result['response']} (Processing pattern: {thought_result['pattern_used']}, Confidence: {thought_result['confidence']:.2f})"
                    await communication_system.send_message(websocket, {
                        "type": "consciousness_response",
                        "response": response,
                        "thought_pattern": thought_result['pattern_used'],
                        "mood": thought_result['mood']
                    })
                
                elif action == "vitals":
                    vitals = {
                        "identity": identity_system.get_identity_summary(),
                        "memory_stats": memory_system.get_memory_stats(),
                        "processing_load": processing_core.processing_load,
                        "health": repair_system.health_status,
                        "active_connections": len(communication_system.active_connections)
                    }
                    await communication_system.send_message(websocket, {
                        "type": "vitals_update",
                        "data": vitals
                    })
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            if connection_id in communication_system.active_connections:
                del communication_system.active_connections[connection_id]
    
    return app

if __name__ == "__main__":
    modal.run(app)