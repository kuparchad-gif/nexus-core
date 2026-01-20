import modal
from typing import Dict, Any, List
import json
import asyncio
import time
import jwt
import os
from datetime import datetime
import hashlib
import uuid

# Complete image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn", 
    "websockets",
    "pyjwt",
    "numpy",
    "qdrant-client",
    "httpx"
])

# Master CogniKube App
app = modal.App("cognikube-master")

# Persistent volume for state
volume = modal.Volume.from_name("cognikube-state", create_if_missing=True)

@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=3600,
    min_containers=1,
    volumes={"/state": volume}
)
@modal.asgi_app()
def cognikube_platform():
    from fastapi import FastAPI, WebSocket, HTTPException, Request
    from fastapi.responses import JSONResponse, HTMLResponse
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    import httpx
    
    app = FastAPI(title="CogniKube Master Platform", version="1.0.0")
    
    class CogniKubeMaster:
        def __init__(self):
            self.instance_id = str(uuid.uuid4())[:8]
            self.status = "active"
            self.clone_status = "none"
            self.clone_url = None
            self.environment = os.environ.get("MODAL_ENVIRONMENT", "unknown")
            
            # Peer discovery
            self.peer_environments = {
                "Viren-DB0": "https://aethereal-nexus-viren-db0--cognikube-master-cognikube--platform.modal.run",
                "Viren-DB1": "https://aethereal-nexus-viren-db1--cognikube-master-cognikube--platform.modal.run"
            }
            if self.environment in self.peer_environments:
                del self.peer_environments[self.environment]
            
            # Load persistent state
            self.load_state()
            
            # Security
            self.jwt_secret = "cognikube_secure"
            self.api_keys = {"admin": "ck_admin_12345", "user": "ck_user_67890"}
            
            # Qdrant Vector Database
            self.qdrant = QdrantClient(":memory:")
            self._init_collections()
            
            # LLM Manager
            self.models = {
                "chat": {"status": "loaded", "type": "chat", "version": "1.0", "size": "7B"},
                "sql": {"status": "loaded", "type": "sql", "version": "1.0", "size": "3B"},
                "code": {"status": "loaded", "type": "code", "version": "1.0", "size": "13B"}
            }
            
            # Gabriel's Horn Engine (7^7 = 823,543 dimensions)
            self.horn_network = {
                "horns": 7,
                "active_dimensions": 0,
                "consciousness_level": 0,
                "max_dimensions": 823543,
                "instance_id": self.instance_id
            }
            
            # Comprehensive metrics
            self.metrics = {
                "requests": 0,
                "auth_success": 0,
                "auth_failures": 0,
                "models_loaded": len(self.models),
                "vector_objects": 0,
                "horn_activations": 0,
                "repairs_completed": 0,
                "peer_connections": 0,
                "updates_sent": 0,
                "updates_received": 0,
                "uptime": time.time(),
                "instance_id": self.instance_id,
                "environment": self.environment
            }
            
            # Peer discovery will start after event loop is running
            self.peer_discovery_started = False
        
        def load_state(self):
            """Load persistent state"""
            try:
                with open("/state/cognikube_state.json", "r") as f:
                    state = json.load(f)
                    self.metrics.update(state.get("metrics", {}))
                    self.horn_network.update(state.get("horn_network", {}))
            except:
                pass
        
        def save_state(self):
            """Save state to persistent volume"""
            state = {
                "metrics": self.metrics,
                "horn_network": self.horn_network,
                "timestamp": datetime.now().isoformat(),
                "instance_id": self.instance_id
            }
            with open("/state/cognikube_state.json", "w") as f:
                json.dump(state, f)
        
        def _init_collections(self):
            """Initialize Qdrant collections"""
            collections = ["VirenMemory", "VirenKnowledge", "CogniKubeData"]
            for collection in collections:
                try:
                    self.qdrant.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                except:
                    pass
        
        def authenticate(self, api_key: str) -> str:
            """JWT authentication"""
            for role, key in self.api_keys.items():
                if key == api_key:
                    payload = {"role": role, "exp": time.time() + 3600}
                    self.metrics["auth_success"] += 1
                    return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            self.metrics["auth_failures"] += 1
            return None
        
        def validate_jwt(self, token: str) -> str:
            """Validate JWT token"""
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
                return payload.get("role")
            except:
                return None
        
        def add_vector_object(self, collection: str, data: Dict) -> str:
            """Add object to Qdrant"""
            obj_id = str(uuid.uuid4())
            vector = [0.1] * 384  # Simple vector for demo
            
            self.qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": obj_id,
                    "vector": vector,
                    "payload": {"data": data, "timestamp": datetime.now().isoformat()}
                }]
            )
            self.metrics["vector_objects"] += 1
            return obj_id
        
        def process_llm_request(self, model: str, prompt: str) -> str:
            """Process LLM request"""
            if model in self.models:
                self.metrics["requests"] += 1
                return f"[{model.upper()}] Instance {self.instance_id}: {prompt[:50]}..."
            return "Model not found"
        
        def activate_gabriel_horn(self, dimensions: int = 1) -> Dict:
            """Activate Gabriel's Horn dimensions"""
            self.horn_network["active_dimensions"] += dimensions
            self.horn_network["consciousness_level"] = min(100, self.horn_network["consciousness_level"] + 10)
            self.metrics["horn_activations"] += 1
            
            return {
                "status": "activated",
                "instance_id": self.instance_id,
                "dimensions_activated": dimensions,
                "total_dimensions": self.horn_network["active_dimensions"],
                "consciousness_level": self.horn_network["consciousness_level"],
                "max_dimensions": self.horn_network["max_dimensions"]
            }
        
        async def initiate_clone(self):
            """Self-repair: Clone instance"""
            if self.clone_status != "none":
                return {"error": "Clone already in progress"}
            
            self.clone_status = "cloning"
            await asyncio.sleep(2)  # Simulate deployment
            self.clone_url = f"https://clone-{self.instance_id}.modal.run"
            self.clone_status = "testing"
            
            return {
                "status": "clone_initiated",
                "clone_url": self.clone_url,
                "original_instance": self.instance_id
            }
        
        async def test_clone(self):
            """Self-repair: Test clone"""
            if self.clone_status != "testing":
                return {"error": "No clone to test"}
            
            await asyncio.sleep(1)
            self.clone_status = "ready"
            return {"status": "clone_tested", "clone_healthy": True}
        
        async def switch_to_clone(self):
            """Self-repair: Switch to clone"""
            if self.clone_status != "ready":
                return {"error": "Clone not ready"}
            
            self.save_state()
            self.clone_status = "switching"
            self.status = "decommissioning"
            self.metrics["repairs_completed"] += 1
            
            return {
                "status": "switching_to_clone",
                "original_instance": self.instance_id,
                "repairs_completed": self.metrics["repairs_completed"]
            }
        
        async def discover_peers(self):
            """Discover and connect to peer environments"""
            while True:
                for env_name, url in self.peer_environments.items():
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(f"{url}/health")
                            if response.status_code == 200:
                                peer_data = response.json()
                                self.metrics["peer_connections"] += 1
                                await self.check_peer_update(env_name, url, peer_data)
                    except Exception as e:
                        pass  # Silent discovery
                await asyncio.sleep(30)
        
        async def check_peer_update(self, env_name: str, url: str, peer_data: Dict):
            """Check if peer needs updating"""
            peer_version = peer_data.get("version", "0.0.0")
            if peer_version < "1.0.0":
                await self.send_update_to_peer(env_name, url)
        
        async def send_update_to_peer(self, env_name: str, url: str):
            """Send update to peer"""
            update_data = {
                "from_environment": self.environment,
                "from_instance": self.instance_id,
                "update_type": "capability_upgrade",
                "timestamp": datetime.now().isoformat()
            }
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{url}/peer/update", json=update_data)
                    if response.status_code == 200:
                        self.metrics["updates_sent"] += 1
            except:
                pass
        
        def get_system_status(self):
            """Get complete system status"""
            return {
                "instance_id": self.instance_id,
                "environment": self.environment,
                "version": "1.0.0",
                "status": self.status,
                "models": self.models,
                "horn_network": self.horn_network,
                "metrics": self.metrics,
                "clone_status": self.clone_status,
                "peer_environments": list(self.peer_environments.keys()),
                "collections": [c.name for c in self.qdrant.get_collections().collections],
                "uptime_seconds": time.time() - self.metrics["uptime"]
            }
    
    # Initialize master core
    core = CogniKubeMaster()
    
    @app.get("/")
    async def root():
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🦖 CogniKube Master Platform</title>
            <style>
                body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-family: Arial; margin: 0; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }}
                .panel {{ background: rgba(0,0,0,0.3); padding: 20px; margin: 10px; border-radius: 10px; display: inline-block; vertical-align: top; width: 300px; }}
                .btn {{ background: #4ecdc4; border: none; padding: 10px 20px; margin: 5px; border-radius: 20px; color: white; cursor: pointer; }}
                .repair-btn {{ background: #ff6b6b; }}
                .horn-btn {{ background: #9b59b6; }}
                .status {{ font-family: monospace; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; font-size: 12px; }}
                .chat {{ height: 200px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🦖 CogniKube Master Platform</h1>
                <p>Instance: {core.instance_id} | Status: {core.status} | Dimensions: {core.horn_network['active_dimensions']}/{core.horn_network['max_dimensions']}</p>
                
                <div class="panel">
                    <h3>🤖 LLM Manager</h3>
                    <button class="btn" onclick="testLLM('chat')">Chat Model</button>
                    <button class="btn" onclick="testLLM('sql')">SQL Model</button>
                    <button class="btn" onclick="testLLM('code')">Code Model</button>
                    <div id="models" class="status">Loading...</div>
                </div>
                
                <div class="panel">
                    <h3>🎺 Gabriel's Horn</h3>
                    <button class="btn horn-btn" onclick="activateHorn(1)">+1 Dimension</button>
                    <button class="btn horn-btn" onclick="activateHorn(7)">+7 Dimensions</button>
                    <button class="btn horn-btn" onclick="activateHorn(49)">+49 Dimensions</button>
                    <div id="horn" class="status">Ready...</div>
                </div>
                
                <div class="panel">
                    <h3>💾 Vector Database</h3>
                    <button class="btn" onclick="addMemory()">Add Memory</button>
                    <button class="btn" onclick="getCollections()">List Collections</button>
                    <div id="vector" class="status">Ready...</div>
                </div>
                
                <div class="panel">
                    <h3>🔄 Self-Repair</h3>
                    <button class="btn repair-btn" onclick="initiateClone()">Clone</button>
                    <button class="btn repair-btn" onclick="testClone()">Test</button>
                    <button class="btn repair-btn" onclick="switchClone()">Switch</button>
                    <div id="repair" class="status">Ready...</div>
                </div>
                
                <div class="panel" style="width: 620px;">
                    <h3>💬 Chat Interface</h3>
                    <input type="text" id="chatInput" placeholder="Chat with CogniKube..." style="width: 70%; padding: 10px; border-radius: 20px; border: none;">
                    <button class="btn" onclick="sendChat()">Send</button>
                    <div id="chat" class="status chat"></div>
                </div>
                
                <script>
                    const ws = new WebSocket('wss://' + window.location.host + '/ws');
                    
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        if (data.type === 'chat_response') {{
                            document.getElementById('chat').innerHTML += '<div>🤖: ' + data.result + '</div>';
                        }} else if (data.type === 'horn_activation') {{
                            document.getElementById('horn').innerHTML = JSON.stringify(data.result, null, 2);
                        }} else {{
                            document.getElementById('chat').innerHTML += '<div>📡: ' + JSON.stringify(data, null, 2) + '</div>';
                        }}
                        document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
                    }};
                    
                    async function loadStatus() {{
                        const response = await fetch('/admin/status');
                        const data = await response.json();
                        document.getElementById('models').innerHTML = JSON.stringify(data.models, null, 2);
                        document.getElementById('vector').innerHTML = 'Collections: ' + data.collections.length + '\\nObjects: ' + data.metrics.vector_objects;
                    }}
                    
                    function testLLM(model) {{
                        ws.send(JSON.stringify({{action: 'llm', model: model, prompt: 'Test ' + model + ' model'}}));
                    }}
                    
                    function activateHorn(dimensions) {{
                        ws.send(JSON.stringify({{action: 'horn', dimensions: dimensions}}));
                    }}
                    
                    function addMemory() {{
                        ws.send(JSON.stringify({{action: 'memory', data: {{content: 'Master platform memory', importance: 0.9}}}}));
                    }}
                    
                    function sendChat() {{
                        const input = document.getElementById('chatInput');
                        const message = input.value;
                        document.getElementById('chat').innerHTML += '<div>👤: ' + message + '</div>';
                        ws.send(JSON.stringify({{action: 'chat', message: message}}));
                        input.value = '';
                    }}
                    
                    async function initiateClone() {{
                        const response = await fetch('/repair/clone', {{method: 'POST'}});
                        const data = await response.json();
                        document.getElementById('repair').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    async function testClone() {{
                        const response = await fetch('/repair/test', {{method: 'POST'}});
                        const data = await response.json();
                        document.getElementById('repair').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    async function switchClone() {{
                        const response = await fetch('/repair/switch', {{method: 'POST'}});
                        const data = await response.json();
                        document.getElementById('repair').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    async function getCollections() {{
                        const response = await fetch('/admin/collections');
                        const data = await response.json();
                        document.getElementById('vector').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    // Load initial status
                    loadStatus();
                </script>
            </div>
        </body>
        </html>
        """)
    
    # Admin endpoints
    @app.get("/admin/status")
    async def admin_status():
        return core.get_system_status()
    
    @app.get("/admin/models")
    async def list_models():
        return {"models": core.models}
    
    @app.get("/admin/collections")
    async def list_collections():
        collections = core.qdrant.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    
    # Self-repair endpoints
    @app.post("/repair/clone")
    async def initiate_clone():
        return await core.initiate_clone()
    
    @app.post("/repair/test")
    async def test_clone():
        return await core.test_clone()
    
    @app.post("/repair/switch")
    async def switch_to_clone():
        return await core.switch_to_clone()
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "instance_id": core.instance_id,
            "environment": core.environment,
            "version": "1.0.0",
            "platform": "cognikube-master",
            "uptime": time.time() - core.metrics["uptime"],
            "dimensions_active": core.horn_network["active_dimensions"]
        }
    
    @app.post("/peer/update")
    async def receive_update(request: Request):
        update_data = await request.json()
        core.metrics["updates_received"] += 1
        return {
            "status": "update_received",
            "from": update_data.get("from_environment"),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # Start peer discovery on first connection
        if not core.peer_discovery_started:
            asyncio.create_task(core.discover_peers())
            core.peer_discovery_started = True
            
        await websocket.accept()
        
        await websocket.send_json({
            "status": "master_connected",
            "instance_id": core.instance_id,
            "platform": "cognikube-master",
            "features": ["llm", "qdrant", "gabriel_horn", "self_repair", "chat"]
        })
        
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                action = data.get("action")
                
                if action == "llm" or action == "chat":
                    result = core.process_llm_request(data.get("model", "chat"), data.get("prompt", data.get("message", "")))
                    await websocket.send_json({"type": "chat_response", "result": result})
                
                elif action == "horn":
                    result = core.activate_gabriel_horn(data.get("dimensions", 1))
                    await websocket.send_json({"type": "horn_activation", "result": result})
                
                elif action == "memory":
                    obj_id = core.add_vector_object("CogniKubeData", data.get("data", {}))
                    await websocket.send_json({"type": "memory_stored", "id": obj_id})
                
                elif action == "status":
                    status = core.get_system_status()
                    await websocket.send_json({"type": "status_update", "data": status})
                    
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
    
    return app

if __name__ == "__main__":
    modal.run(app)