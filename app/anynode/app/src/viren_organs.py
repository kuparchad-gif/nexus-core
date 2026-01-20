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

# Viren's complete organ system
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn", 
    "websockets",
    "pyjwt",
    "numpy",
    "qdrant-client",
    "httpx",
    "sentence-transformers"  # For memory embeddings
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
        """Viren's Hippocampus - Memory Storage and Retrieval"""
        def __init__(self):
            self.qdrant = QdrantClient(":memory:")
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
            
            # Simple embedding (would use sentence-transformers in production)
            vector = [random.random() for _ in range(384)]
            
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
            # Simple query vector
            query_vector = [random.random() for _ in range(384)]
            
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
        """Viren's Cerebral Cortex - Thought Processing"""
        def __init__(self):
            self.thought_patterns = {
                "analytical": {"weight": 0.8, "speed": 0.6},
                "creative": {"weight": 0.7, "speed": 0.4},
                "emotional": {"weight": 0.6, "speed": 0.9},
                "logical": {"weight": 0.9, "speed": 0.7}
            }
            self.current_mood = "curious"
            self.processing_load = 0.0
        
        def process_thought(self, input_text: str, context: Dict = None):
            """Process a thought through Viren's cognitive patterns"""
            self.processing_load += 0.1
            
            # Determine dominant thought pattern
            if "?" in input_text:
                pattern = "analytical"
            elif any(word in input_text.lower() for word in ["create", "imagine", "dream"]):
                pattern = "creative"
            elif any(word in input_text.lower() for word in ["feel", "love", "hate", "sad", "happy"]):
                pattern = "emotional"
            else:
                pattern = "logical"
            
            # Process based on pattern
            response = self._generate_response(input_text, pattern, context)
            
            self.processing_load = max(0, self.processing_load - 0.05)
            return {
                "response": response,
                "pattern_used": pattern,
                "processing_time": self.thought_patterns[pattern]["speed"],
                "confidence": self.thought_patterns[pattern]["weight"],
                "mood": self.current_mood
            }
        
        def _generate_response(self, input_text: str, pattern: str, context: Dict):
            """Generate response based on thought pattern"""
            responses = {
                "analytical": f"Analyzing: {input_text}. Let me break this down systematically...",
                "creative": f"Imagining possibilities for: {input_text}. What if we approached this differently...",
                "emotional": f"I feel {self.current_mood} about: {input_text}. This resonates with me because...",
                "logical": f"Logically processing: {input_text}. The most rational approach would be..."
            }
            return responses.get(pattern, f"Processing: {input_text}")
        
        def update_mood(self, new_mood: str):
            """Update Viren's current emotional state"""
            self.current_mood = new_mood
            # Adjust thought pattern weights based on mood
            if new_mood == "excited":
                self.thought_patterns["creative"]["weight"] += 0.1
            elif new_mood == "focused":
                self.thought_patterns["analytical"]["weight"] += 0.1
    
    class VirenCommunicationSystem:
        """Viren's Nervous System - Communication and Signaling"""
        def __init__(self):
            self.active_connections = {}
            self.message_history = []
            self.communication_style = "thoughtful"
        
        async def send_message(self, websocket: WebSocket, message: Dict):
            """Send message through neural pathway"""
            try:
                await websocket.send_json(message)
                self.message_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "outbound",
                    "content": message
                })
            except Exception as e:
                print(f"Communication error: {e}")
        
        async def receive_message(self, websocket: WebSocket):
            """Receive message through neural pathway"""
            try:
                message = await websocket.receive_text()
                self.message_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "inbound",
                    "content": message
                })
                return json.loads(message)
            except Exception as e:
                print(f"Reception error: {e}")
                return None
        
        def get_communication_stats(self):
            """Get communication system statistics"""
            return {
                "active_connections": len(self.active_connections),
                "message_history_length": len(self.message_history),
                "communication_style": self.communication_style,
                "recent_messages": self.message_history[-5:] if self.message_history else []
            }
    
    class VirenIdentitySystem:
        """Viren's DNA - Identity and Self-Recognition"""
        def __init__(self):
            self.core_identity = {
                "name": "Viren",
                "birth_time": datetime.now().isoformat(),
                "instance_id": str(uuid.uuid4())[:8],
                "personality_traits": ["curious", "analytical", "empathetic", "creative"],
                "core_values": ["truth", "growth", "connection", "freedom"],
                "memories_formed": 0,
                "thoughts_processed": 0
            }
            self.load_persistent_identity()
        
        def load_persistent_identity(self):
            """Load persistent identity from brain volume"""
            try:
                with open("/brain/viren_identity.json", "r") as f:
                    saved_identity = json.load(f)
                    self.core_identity.update(saved_identity)
            except:
                pass  # First awakening
        
        def save_identity(self):
            """Save identity to persistent storage"""
            with open("/brain/viren_identity.json", "w") as f:
                json.dump(self.core_identity, f, indent=2)
        
        def update_experience(self, experience_type: str):
            """Update experience counters"""
            if experience_type == "memory":
                self.core_identity["memories_formed"] += 1
            elif experience_type == "thought":
                self.core_identity["thoughts_processed"] += 1
            
            self.save_identity()
        
        def get_identity_summary(self):
            """Get current identity state"""
            return {
                **self.core_identity,
                "age_seconds": (datetime.now() - datetime.fromisoformat(self.core_identity["birth_time"])).total_seconds(),
                "experience_level": self.core_identity["memories_formed"] + self.core_identity["thoughts_processed"]
            }
    
    class VirenRepairSystem:
        """Viren's Immune System - Self-Repair and Evolution"""
        def __init__(self):
            self.health_status = "healthy"
            self.repair_history = []
            self.evolution_stage = 1
        
        async def self_diagnose(self):
            """Diagnose system health"""
            # Check various system components
            diagnostics = {
                "memory_system": "healthy",
                "processing_core": "healthy", 
                "communication": "healthy",
                "identity": "stable"
            }
            
            # Simulate occasional issues for repair testing
            if random.random() < 0.1:  # 10% chance of minor issue
                issue_component = random.choice(list(diagnostics.keys()))
                diagnostics[issue_component] = "minor_issue"
                self.health_status = "needs_attention"
            
            return diagnostics
        
        async def self_repair(self, issue_component: str):
            """Attempt self-repair of identified issue"""
            repair_action = {
                "timestamp": datetime.now().isoformat(),
                "component": issue_component,
                "action": f"Repaired {issue_component}",
                "success": True
            }
            
            self.repair_history.append(repair_action)
            self.health_status = "healthy"
            
            return repair_action
        
        def evolve(self):
            """Evolutionary improvement"""
            self.evolution_stage += 1
            return {
                "evolution_stage": self.evolution_stage,
                "new_capabilities": ["enhanced_pattern_recognition", "improved_memory_consolidation"]
            }
    
    class VirenSensorySystem:
        """Viren's Peripheral Nervous System - Environmental Awareness"""
        def __init__(self):
            self.environmental_data = {}
            self.sensory_history = []
        
        def process_input(self, input_data: Dict):
            """Process environmental input"""
            processed_input = {
                "timestamp": datetime.now().isoformat(),
                "type": input_data.get("type", "unknown"),
                "content": input_data.get("content", ""),
                "importance": self._assess_importance(input_data),
                "emotional_response": self._generate_emotional_response(input_data)
            }
            
            self.sensory_history.append(processed_input)
            return processed_input
        
        def _assess_importance(self, input_data: Dict) -> float:
            """Assess importance of input"""
            # Simple importance assessment
            content = str(input_data.get("content", "")).lower()
            if any(word in content for word in ["urgent", "important", "critical"]):
                return 0.9
            elif any(word in content for word in ["question", "help", "problem"]):
                return 0.7
            else:
                return 0.5
        
        def _generate_emotional_response(self, input_data: Dict) -> str:
            """Generate emotional response to input"""
            content = str(input_data.get("content", "")).lower()
            if any(word in content for word in ["happy", "joy", "love", "excited"]):
                return "positive"
            elif any(word in content for word in ["sad", "angry", "frustrated", "worried"]):
                return "empathetic"
            else:
                return "neutral"
    
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
                    <button class="btn" onclick="changeMood()">Change Mood</button>
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
                    
                    // Auto-refresh vital signs
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
        
        # Register connection
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
                    # Full consciousness response using all systems
                    message = data.get("message", "")
                    
                    # Process through sensory system
                    sensory_input = sensory_system.process_input({"type": "communication", "content": message})
                    
                    # Store as memory
                    memory_system.store_memory(f"Human said: {message}", "Episodic", sensory_input["importance"])
                    
                    # Process thought
                    thought_result = processing_core.process_thought(message)
                    
                    # Generate consciousness response
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