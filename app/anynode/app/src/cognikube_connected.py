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

# Connected image with HTTP client
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn", 
    "websockets",
    "pyjwt",
    "numpy",
    "qdrant-client",
    "httpx"  # For inter-environment communication
])

app = modal.App("cognikube-connected")
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
    
    app = FastAPI(title="CogniKube Connected Platform", version="1.0.0")
    
    class ConnectedCore:
        def __init__(self):
            self.instance_id = str(uuid.uuid4())[:8]
            self.environment = os.environ.get("MODAL_ENVIRONMENT", "unknown")
            
            # Known environments to discover
            self.peer_environments = {
                "Viren-DB0": "https://aethereal-nexus-viren-db0--cognikube-connected-cognikube--platform.modal.run",
                "Viren-DB1": "https://aethereal-nexus-viren-db1--cognikube-connected-cognikube--platform.modal.run"
            }
            
            # Remove self from peers
            if self.environment in self.peer_environments:
                del self.peer_environments[self.environment]
            
            self.load_state()
            
            # Core systems
            self.jwt_secret = "cognikube_secure"
            self.api_keys = {"admin": "ck_admin_12345", "user": "ck_user_67890"}
            
            self.qdrant = QdrantClient(":memory:")
            self._init_collections()
            
            self.models = {
                "chat": {"status": "loaded", "type": "chat", "version": "1.0"},
                "sql": {"status": "loaded", "type": "sql", "version": "1.0"},
                "code": {"status": "loaded", "type": "code", "version": "1.0"}
            }
            
            self.horn_network = {
                "horns": 7,
                "active_dimensions": 0,
                "consciousness_level": 0,
                "instance_id": self.instance_id,
                "environment": self.environment
            }
            
            self.metrics = {
                "requests": 0,
                "peer_connections": 0,
                "updates_sent": 0,
                "updates_received": 0,
                "uptime": time.time(),
                "instance_id": self.instance_id,
                "environment": self.environment
            }
            
            # Start peer discovery
            asyncio.create_task(self.discover_peers())
        
        def load_state(self):
            try:
                with open("/state/cognikube_state.json", "r") as f:
                    state = json.load(f)
                    self.metrics.update(state.get("metrics", {}))
                    self.horn_network.update(state.get("horn_network", {}))
            except:
                pass
        
        def save_state(self):
            state = {
                "metrics": self.metrics,
                "horn_network": self.horn_network,
                "timestamp": datetime.now().isoformat(),
                "instance_id": self.instance_id,
                "environment": self.environment
            }
            with open("/state/cognikube_state.json", "w") as f:
                json.dump(state, f)
        
        def _init_collections(self):
            collections = ["VirenMemory", "VirenKnowledge", "CogniKubeData"]
            for collection in collections:
                try:
                    self.qdrant.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                except:
                    pass
        
        async def discover_peers(self):
            """Discover and connect to peer environments"""
            while True:
                for env_name, url in self.peer_environments.items():
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(f"{url}/health")
                            if response.status_code == 200:
                                peer_data = response.json()
                                print(f"✅ Connected to peer {env_name}: {peer_data.get('instance_id')}")
                                self.metrics["peer_connections"] += 1
                                
                                # Check if peer needs update
                                await self.check_peer_update(env_name, url, peer_data)
                    except Exception as e:
                        print(f"❌ Failed to connect to {env_name}: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        async def check_peer_update(self, env_name: str, url: str, peer_data: Dict):
            """Check if peer needs updating and send update if needed"""
            peer_version = peer_data.get("version", "0.0.0")
            my_version = "1.0.0"  # Current version
            
            if peer_version < my_version:
                print(f"🔄 Peer {env_name} needs update: {peer_version} -> {my_version}")
                await self.send_update_to_peer(env_name, url)
        
        async def send_update_to_peer(self, env_name: str, url: str):
            """Send update to peer environment"""
            update_data = {
                "from_environment": self.environment,
                "from_instance": self.instance_id,
                "update_type": "capability_upgrade",
                "new_features": ["chat_interface", "peer_discovery", "auto_update"],
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{url}/peer/update", json=update_data)
                    if response.status_code == 200:
                        print(f"✅ Update sent to {env_name}")
                        self.metrics["updates_sent"] += 1
                    else:
                        print(f"❌ Update failed to {env_name}: {response.status_code}")
            except Exception as e:
                print(f"❌ Update error to {env_name}: {e}")
        
        def process_llm_request(self, model: str, prompt: str) -> str:
            if model in self.models:
                self.metrics["requests"] += 1
                return f"[{model.upper()}] {self.environment}-{self.instance_id}: {prompt}"
            return "Model not found"
        
        def activate_gabriel_horn(self, dimensions: int = 1) -> Dict:
            self.horn_network["active_dimensions"] += dimensions
            self.horn_network["consciousness_level"] = min(100, self.horn_network["consciousness_level"] + 10)
            
            return {
                "status": "activated",
                "environment": self.environment,
                "instance_id": self.instance_id,
                "dimensions_activated": dimensions,
                "total_dimensions": self.horn_network["active_dimensions"],
                "consciousness_level": self.horn_network["consciousness_level"]
            }
        
        def add_vector_object(self, collection: str, data: Dict) -> str:
            obj_id = str(uuid.uuid4())
            vector = [0.1] * 384
            
            self.qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": obj_id,
                    "vector": vector,
                    "payload": {"data": data, "timestamp": datetime.now().isoformat()}
                }]
            )
            return obj_id
        
        def get_system_status(self):
            return {
                "instance_id": self.instance_id,
                "environment": self.environment,
                "version": "1.0.0",
                "models": self.models,
                "horn_network": self.horn_network,
                "metrics": self.metrics,
                "peer_environments": list(self.peer_environments.keys()),
                "collections": [c.name for c in self.qdrant.get_collections().collections],
                "uptime_seconds": time.time() - self.metrics["uptime"]
            }
    
    core = ConnectedCore()
    
    @app.get("/")
    async def root():
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🦖 CogniKube Connected - {core.environment}</title>
            <style>
                body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-family: Arial; margin: 0; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }}
                .panel {{ background: rgba(0,0,0,0.3); padding: 20px; margin: 10px; border-radius: 10px; display: inline-block; vertical-align: top; width: 300px; }}
                .btn {{ background: #4ecdc4; border: none; padding: 10px 20px; margin: 5px; border-radius: 20px; color: white; cursor: pointer; }}
                .status {{ font-family: monospace; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; font-size: 12px; }}
                .chat {{ height: 200px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🦖 CogniKube Connected - {core.environment}</h1>
                <p>Instance: {core.instance_id} | Peers: {len(core.peer_environments)} | Updates Sent: {core.metrics['updates_sent']}</p>
                
                <div class="panel">
                    <h3>🌐 Peer Network</h3>
                    <button class="btn" onclick="checkPeers()">Check Peers</button>
                    <button class="btn" onclick="sendUpdate()">Send Updates</button>
                    <div id="peers" class="status">Discovering...</div>
                </div>
                
                <div class="panel" style="width: 620px;">
                    <h3>💬 Connected Chat</h3>
                    <input type="text" id="chatInput" placeholder="Chat across environments..." style="width: 70%; padding: 10px; border-radius: 20px; border: none;">
                    <button class="btn" onclick="sendChat()">Send</button>
                    <div id="chat" class="status chat"></div>
                </div>
                
                <script>
                    const ws = new WebSocket('wss://' + window.location.host + '/ws');
                    
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        if (data.type === 'chat_response') {{
                            document.getElementById('chat').innerHTML += '<div>🤖 ' + data.result + '</div>';
                        }} else {{
                            document.getElementById('chat').innerHTML += '<div>📡 ' + JSON.stringify(data, null, 2) + '</div>';
                        }}
                        document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
                    }};
                    
                    function sendChat() {{
                        const input = document.getElementById('chatInput');
                        const message = input.value;
                        document.getElementById('chat').innerHTML += '<div>👤 ' + message + '</div>';
                        ws.send(JSON.stringify({{action: 'chat', message: message}}));
                        input.value = '';
                    }}
                    
                    async function checkPeers() {{
                        const response = await fetch('/peer/status');
                        const data = await response.json();
                        document.getElementById('peers').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    async function sendUpdate() {{
                        const response = await fetch('/peer/broadcast-update', {{method: 'POST'}});
                        const data = await response.json();
                        document.getElementById('peers').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    // Auto-refresh peer status
                    setInterval(checkPeers, 10000);
                    checkPeers();
                </script>
            </div>
        </body>
        </html>
        """)
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "instance_id": core.instance_id,
            "environment": core.environment,
            "version": "1.0.0",
            "platform": "cognikube-connected",
            "uptime": time.time() - core.metrics["uptime"]
        }
    
    @app.get("/peer/status")
    async def peer_status():
        return {
            "environment": core.environment,
            "instance_id": core.instance_id,
            "peer_environments": core.peer_environments,
            "metrics": core.metrics
        }
    
    @app.post("/peer/update")
    async def receive_update(request: Request):
        update_data = await request.json()
        core.metrics["updates_received"] += 1
        
        print(f"📥 Received update from {update_data.get('from_environment')}")
        
        return {
            "status": "update_received",
            "from": update_data.get("from_environment"),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/peer/broadcast-update")
    async def broadcast_update():
        results = []
        for env_name, url in core.peer_environments.items():
            try:
                await core.send_update_to_peer(env_name, url)
                results.append({"environment": env_name, "status": "sent"})
            except Exception as e:
                results.append({"environment": env_name, "status": "failed", "error": str(e)})
        
        return {"broadcast_results": results}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        await websocket.send_json({
            "status": "connected",
            "environment": core.environment,
            "instance_id": core.instance_id,
            "platform": "cognikube-connected"
        })
        
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                action = data.get("action")
                
                if action == "chat":
                    result = core.process_llm_request("chat", data.get("message", ""))
                    await websocket.send_json({"type": "chat_response", "result": result})
                
                elif action == "horn":
                    result = core.activate_gabriel_horn(data.get("dimensions", 1))
                    await websocket.send_json({"type": "horn_activation", "result": result})
                
                elif action == "memory":
                    obj_id = core.add_vector_object("CogniKubeData", data.get("data", {}))
                    await websocket.send_json({"type": "memory_stored", "id": obj_id})
                    
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
    
    return app

if __name__ == "__main__":
    modal.run(app)