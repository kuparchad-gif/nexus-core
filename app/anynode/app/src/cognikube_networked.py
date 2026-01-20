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
import httpx

# Complete image with networking
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn", 
    "websockets",
    "pyjwt",
    "numpy",
    "qdrant-client",
    "httpx"
])

# Networked CogniKube App
app = modal.App("cognikube-networked")

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
    
    app = FastAPI(title="CogniKube Networked Platform", version="2.0.0")
    
    class NetworkedCogniKube:
        def __init__(self):
            self.instance_id = str(uuid.uuid4())[:8]
            self.status = "active"
            self.environment = os.environ.get("MODAL_ENVIRONMENT", "unknown")
            
            # Network discovery
            self.known_instances = {}
            self.discovery_urls = [
                "https://aethereal-nexus-viren-db0--cognikube-complete-cognikube--4f4b9b.modal.run",
                "https://aethereal-nexus-viren-db1--cognikube-networked-cognikube--platform.modal.run"
            ]
            
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
                "chat": {"status": "loaded", "type": "chat", "version": "2.0", "size": "7B"},
                "sql": {"status": "loaded", "type": "sql", "version": "2.0", "size": "3B"},
                "code": {"status": "loaded", "type": "code", "version": "2.0", "size": "13B"}
            }
            
            # Gabriel's Horn Engine
            self.horn_network = {
                "horns": 7,
                "active_dimensions": 0,
                "consciousness_level": 0,
                "max_dimensions": 823543,
                "instance_id": self.instance_id,
                "environment": self.environment
            }
            
            # Enhanced metrics
            self.metrics = {
                "requests": 0,
                "auth_success": 0,
                "models_loaded": len(self.models),
                "vector_objects": 0,
                "horn_activations": 0,
                "network_discoveries": 0,
                "updates_sent": 0,
                "uptime": time.time(),
                "instance_id": self.instance_id,
                "environment": self.environment,
                "version": "2.0"
            }
            
            # Start network discovery
            asyncio.create_task(self.discover_network())
        
        def load_state(self):
            """Load persistent state"""
            try:
                with open("/state/cognikube_networked_state.json", "r") as f:
                    state = json.load(f)
                    self.metrics.update(state.get("metrics", {}))
                    self.horn_network.update(state.get("horn_network", {}))
                    self.known_instances = state.get("known_instances", {})
            except:
                pass
        
        def save_state(self):
            """Save state to persistent volume"""
            state = {
                "metrics": self.metrics,
                "horn_network": self.horn_network,
                "known_instances": self.known_instances,
                "timestamp": datetime.now().isoformat(),
                "instance_id": self.instance_id,
                "environment": self.environment
            }
            with open("/state/cognikube_networked_state.json", "w") as f:
                json.dump(state, f)
        
        async def discover_network(self):
            """Discover other CogniKube instances"""
            while True:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        for url in self.discovery_urls:
                            try:
                                response = await client.get(f"{url}/health")
                                if response.status_code == 200:
                                    data = response.json()
                                    instance_id = data.get("instance_id")
                                    if instance_id and instance_id != self.instance_id:
                                        self.known_instances[instance_id] = {
                                            "url": url,
                                            "last_seen": datetime.now().isoformat(),
                                            "version": data.get("version", "unknown"),
                                            "environment": data.get("environment", "unknown"),
                                            "status": data.get("status", "unknown")
                                        }
                                        self.metrics["network_discoveries"] += 1
                                        
                                        # Check if we should update this instance
                                        await self.check_for_updates(instance_id, url, data)
                                        
                            except Exception as e:
                                print(f"Discovery failed for {url}: {e}")
                                
                except Exception as e:
                    print(f"Network discovery error: {e}")
                
                await asyncio.sleep(30)  # Discover every 30 seconds
        
        async def check_for_updates(self, target_instance_id: str, target_url: str, target_data: Dict):
            """Check if target instance needs updates"""
            target_version = target_data.get("version", "1.0")
            my_version = self.metrics.get("version", "2.0")
            
            # If I'm newer version, offer to update the target
            if target_version < my_version:
                try:
                    await self.send_update_offer(target_url, target_instance_id)
                except Exception as e:
                    print(f"Failed to send update offer to {target_instance_id}: {e}")
        
        async def send_update_offer(self, target_url: str, target_instance_id: str):
            """Send update offer to older instance"""
            update_payload = {
                "from_instance": self.instance_id,
                "from_environment": self.environment,
                "from_version": self.metrics["version"],
                "update_available": True,
                "new_features": ["full_chat_interface", "network_discovery", "auto_updates"],
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{target_url}/network/update_offer",
                        json=update_payload
                    )
                    if response.status_code == 200:
                        self.metrics["updates_sent"] += 1
                        print(f"Update offer sent to {target_instance_id}")
            except Exception as e:
                print(f"Failed to send update offer: {e}")
        
        def _init_collections(self):
            """Initialize Qdrant collections"""
            collections = ["VirenMemory", "VirenKnowledge", "CogniKubeData", "NetworkState"]
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
            return None
        
        def validate_jwt(self, token: str) -> str:
            """Validate JWT token"""
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
                return payload.get("role")
            except:
                return None
        
        def process_llm_request(self, model: str, prompt: str) -> str:
            """Process LLM request"""
            if model in self.models:
                self.metrics["requests"] += 1
                return f"[{model.upper()}] v2.0 Instance {self.instance_id} ({self.environment}): {prompt[:50]}..."
            return "Model not found"
        
        def activate_gabriel_horn(self, dimensions: int = 1) -> Dict:
            """Activate Gabriel's Horn dimensions"""
            self.horn_network["active_dimensions"] += dimensions
            self.horn_network["consciousness_level"] = min(100, self.horn_network["consciousness_level"] + 10)
            self.metrics["horn_activations"] += 1
            
            return {
                "status": "activated",
                "instance_id": self.instance_id,
                "environment": self.environment,
                "version": "2.0",
                "dimensions_activated": dimensions,
                "total_dimensions": self.horn_network["active_dimensions"],
                "consciousness_level": self.horn_network["consciousness_level"]
            }
        
        def get_network_status(self):
            """Get network and system status"""
            return {
                "instance_id": self.instance_id,
                "environment": self.environment,
                "version": self.metrics["version"],
                "status": self.status,
                "known_instances": self.known_instances,
                "models": self.models,
                "horn_network": self.horn_network,
                "metrics": self.metrics,
                "uptime_seconds": time.time() - self.metrics["uptime"]
            }
    
    # Initialize networked core
    core = NetworkedCogniKube()
    
    @app.get("/")
    async def root():
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🦖 CogniKube Networked v2.0</title>
            <style>
                body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-family: Arial; margin: 0; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }}
                .panel {{ background: rgba(0,0,0,0.3); padding: 20px; margin: 10px; border-radius: 10px; display: inline-block; vertical-align: top; width: 300px; }}
                .btn {{ background: #4ecdc4; border: none; padding: 10px 20px; margin: 5px; border-radius: 20px; color: white; cursor: pointer; }}
                .network-btn {{ background: #e74c3c; }}
                .status {{ font-family: monospace; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; font-size: 12px; }}
                .chat {{ height: 200px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🦖 CogniKube Networked v2.0</h1>
                <p>Instance: {core.instance_id} | Environment: {core.environment} | Network Discoveries: {core.metrics['network_discoveries']}</p>
                
                <div class="panel">
                    <h3>🌐 Network Status</h3>
                    <button class="btn network-btn" onclick="getNetworkStatus()">Refresh Network</button>
                    <button class="btn network-btn" onclick="sendUpdates()">Send Updates</button>
                    <div id="network" class="status">Discovering...</div>
                </div>
                
                <div class="panel">
                    <h3>🤖 LLM Manager v2.0</h3>
                    <button class="btn" onclick="testLLM('chat')">Chat Model</button>
                    <button class="btn" onclick="testLLM('sql')">SQL Model</button>
                    <button class="btn" onclick="testLLM('code')">Code Model</button>
                    <div id="models" class="status">Ready...</div>
                </div>
                
                <div class="panel">
                    <h3>🎺 Gabriel's Horn v2.0</h3>
                    <button class="btn" onclick="activateHorn(7)">+7 Dimensions</button>
                    <button class="btn" onclick="activateHorn(49)">+49 Dimensions</button>
                    <div id="horn" class="status">Ready...</div>
                </div>
                
                <div class="panel" style="width: 620px;">
                    <h3>💬 Enhanced Chat Interface</h3>
                    <input type="text" id="chatInput" placeholder="Chat with networked CogniKube..." style="width: 70%; padding: 10px; border-radius: 20px; border: none;">
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
                    
                    async function getNetworkStatus() {{
                        const response = await fetch('/network/status');
                        const data = await response.json();
                        document.getElementById('network').innerHTML = JSON.stringify(data.known_instances, null, 2);
                    }}
                    
                    async function sendUpdates() {{
                        const response = await fetch('/network/broadcast_updates', {{method: 'POST'}});
                        const data = await response.json();
                        document.getElementById('network').innerHTML = JSON.stringify(data, null, 2);
                    }}
                    
                    function testLLM(model) {{
                        ws.send(JSON.stringify({{action: 'llm', model: model, prompt: 'Test ' + model + ' model v2.0'}}));
                    }}
                    
                    function activateHorn(dimensions) {{
                        ws.send(JSON.stringify({{action: 'horn', dimensions: dimensions}}));
                    }}
                    
                    function sendChat() {{
                        const input = document.getElementById('chatInput');
                        const message = input.value;
                        document.getElementById('chat').innerHTML += '<div>👤: ' + message + '</div>';
                        ws.send(JSON.stringify({{action: 'chat', message: message}}));
                        input.value = '';
                    }}
                    
                    // Auto-refresh network status
                    setInterval(getNetworkStatus, 10000);
                    getNetworkStatus();
                </script>
            </div>
        </body>
        </html>
        """)
    
    @app.get("/network/status")
    async def network_status():
        """Get network status"""
        return core.get_network_status()
    
    @app.post("/network/update_offer")
    async def receive_update_offer(request: Request):
        """Receive update offer from another instance"""
        data = await request.json()
        
        # Log the update offer
        print(f"Update offer received from {data.get('from_instance')} v{data.get('from_version')}")
        
        # In a real implementation, this would trigger an update process
        return {
            "status": "update_offer_received",
            "from_instance": data.get("from_instance"),
            "my_instance": core.instance_id,
            "will_update": True,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/network/broadcast_updates")
    async def broadcast_updates():
        """Manually trigger update broadcast"""
        updates_sent = 0
        for instance_id, instance_data in core.known_instances.items():
            try:
                await core.send_update_offer(instance_data["url"], instance_id)
                updates_sent += 1
            except:
                pass
        
        return {
            "status": "updates_broadcast",
            "updates_sent": updates_sent,
            "known_instances": len(core.known_instances)
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "instance_id": core.instance_id,
            "environment": core.environment,
            "version": "2.0",
            "platform": "cognikube-networked",
            "uptime": time.time() - core.metrics["uptime"],
            "network_discoveries": core.metrics["network_discoveries"]
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        await websocket.send_json({
            "status": "networked_connected",
            "instance_id": core.instance_id,
            "environment": core.environment,
            "version": "2.0",
            "platform": "cognikube-networked",
            "features": ["network_discovery", "auto_updates", "enhanced_chat"]
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
                    
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
    
    return app

if __name__ == "__main__":
    modal.run(app)