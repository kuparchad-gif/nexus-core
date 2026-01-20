#!/usr/bin/env python3
"""
Enhanced CogniKube Deployment
Deploys the enhanced Gabriel's Horn network with real LLM integration and Loki monitoring
"""
import modal
import os
import sys
import asyncio
import json
import time
from typing import Dict, Any, List

# Create Modal app
app = modal.App("cognikube-enhanced")

# Create image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "websockets==12.0",
    "pydantic==2.5.0",
    "cryptography==41.0.7",
    "pyjwt==2.8.0",
    "httpx==0.23.0",
    "requests==2.31.0",
    "numpy==1.26.1",
    "pandas==2.0.3",
    "matplotlib==3.7.2",
    "transformers==4.35.0",
    "huggingface-hub==0.16.4"
])

# Create volume for persistence
volume = modal.Volume.from_name("cognikube-enhanced-brain", create_if_missing=True)

@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/brain": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.asgi_app()
def cognikube_app():
    """Enhanced CogniKube application with real LLM integration and Loki monitoring"""
    # Import all modules here to ensure they're available in the Modal container
    from fastapi import FastAPI, WebSocket, HTTPException, Request, BackgroundTasks
    from fastapi.responses import JSONResponse, HTMLResponse
    import asyncio
    import json
    import time
    import os
    import uuid
    from typing import Dict, Any, List, Optional
    
    # Define the LLM Manager
    class LLMManager:
        def __init__(self, api_key=None):
            self.api_key = api_key or os.environ.get("HF_TOKEN")
            self.base_url = "https://api-inference.huggingface.co/models"
            self.models = {
                "gemma-2b": "google/gemma-2b",
                "hermes-2-pro-llama-3-7b": "NousResearch/hermes-2-pro-llama-3-7b",
                "qwen2.5-14b": "Qwen/Qwen2.5-14B"
            }
            self.cache = {}
            self.last_call = {}
            self.rate_limit_delay = 1.0
        
        async def generate(self, prompt, model="gemma-2b", max_tokens=256, temperature=0.7):
            import httpx
            
            # Check cache
            cache_key = f"{model}:{prompt}:{max_tokens}:{temperature}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Rate limiting
            if model in self.last_call:
                elapsed = time.time() - self.last_call[model]
                if elapsed < self.rate_limit_delay:
                    await asyncio.sleep(self.rate_limit_delay - elapsed)
            
            # Get model ID
            model_id = self.models.get(model)
            if not model_id:
                return {"error": f"Unsupported model: {model}", "text": ""}
            
            # Prepare request
            url = f"{self.base_url}/{model_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            
            try:
                # Make API call
                async with httpx.AsyncClient(timeout=30.0) as client:
                    self.last_call[model] = time.time()
                    response = await client.post(url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Process result
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get("generated_text", "")
                        else:
                            text = result.get("generated_text", "")
                        
                        output = {
                            "text": text,
                            "model": model,
                            "prompt": prompt,
                            "timestamp": time.time()
                        }
                        
                        # Cache result
                        self.cache[cache_key] = output
                        
                        return output
                    else:
                        error_msg = f"API Error: {response.status_code} - {response.text}"
                        print(error_msg)
                        
                        # Try fallback model if available
                        if model != "gemma-2b":
                            print(f"Falling back to gemma-2b")
                            return await self.generate(prompt, "gemma-2b", max_tokens, temperature)
                        
                        return {"error": error_msg, "text": ""}
            
            except Exception as e:
                error_msg = f"Request Error: {str(e)}"
                print(error_msg)
                return {"error": error_msg, "text": ""}
        
        async def get_routing_decision(self, station_id, message, connections, routing_history=None):
            # Prepare prompt
            prompt = f"""
You are station {station_id}, a routing node in the Gabriel's Horn network.
Your job is to route messages efficiently to their destination.

Message: {json.dumps(message, indent=2)}

Your connections:
{json.dumps(connections, indent=2)}

"""
            
            if routing_history:
                prompt += f"""
Recent routing history:
{json.dumps(routing_history, indent=2)}

"""
            
            prompt += """
Which station should I route this message to? Respond with just the station ID.
"""
            
            # Get model based on station_id
            model = "gemma-2b"
            if "horn2" in station_id or "pod2" in station_id:
                model = "hermes-2-pro-llama-3-7b"
            elif "horn3" in station_id or "pod3" in station_id:
                model = "qwen2.5-14b"
            
            # Generate response
            result = await self.generate(prompt, model, max_tokens=32, temperature=0.3)
            
            # Extract station ID from response
            response_text = result.get("text", "").strip()
            
            # Check if response matches any connection
            for connection in connections:
                if connection["id"] in response_text:
                    return connection["id"]
            
            # Fallback to numerical proximity
            destination_value = message.get("destination_value", 0)
            closest_station = None
            min_distance = float('infinity')
            
            for station in connections:
                distance = abs(station["value"] - destination_value)
                if distance < min_distance:
                    min_distance = distance
                    closest_station = station["id"]
            
            return closest_station
    
    # Define the Loki Observer
    class LokiObserver:
        def __init__(self, loki_url="http://localhost:3100"):
            self.loki_url = loki_url
            self.push_endpoint = f"{loki_url}/loki/api/v1/push"
            self.query_endpoint = f"{loki_url}/loki/api/v1/query_range"
            
            # Default labels
            self.default_labels = {
                "application": "cognikube",
                "environment": os.environ.get("ENVIRONMENT", "development"),
                "version": os.environ.get("VERSION", "1.0.0")
            }
        
        def log_event(self, labels, message, level="info"):
            # Merge with default labels
            all_labels = {**self.default_labels, **labels, "level": level}
            
            # Use current time
            ts = str(int(time.time() * 1000000000))  # nanoseconds
            
            payload = {
                "streams": [
                    {
                        "stream": all_labels,
                        "values": [[ts, message]]
                    }
                ]
            }
            
            try:
                import requests
                requests.post(self.push_endpoint, json=payload)
                return True
            except Exception as e:
                print(f"Loki logging error: {e}")
                return False
        
        def query_logs(self, query_string, start_time=None, end_time=None, limit=100):
            if not start_time:
                start_time = int((time.time() - 3600) * 1000000000)  # 1 hour ago
            if not end_time:
                end_time = int(time.time() * 1000000000)  # now
            
            # Prepare query parameters
            params = {
                "query": query_string,
                "start": start_time,
                "end": end_time,
                "limit": limit
            }
            
            try:
                import requests
                response = requests.get(self.query_endpoint, params=params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Loki query error: {response.status_code} - {response.text}")
                    return {"data": {"result": []}}
            except Exception as e:
                print(f"Loki query error: {e}")
                return {"data": {"result": []}}
    
    # Define the LLM Station
    class LLMStation:
        def __init__(self, station_id, station_value, model_name="gemma-2b", llm_manager=None, loki=None):
            self.station_id = station_id
            self.station_value = station_value
            self.model_name = model_name
            self.connections = []
            self.llm_manager = llm_manager
            self.loki = loki
            
            # Log station creation
            if self.loki:
                self.loki.log_event(
                    {"component": "station", "station": station_id, "action": "create", "model": model_name},
                    f"Created station {station_id} with value {station_value} using model {model_name}"
                )
        
        def connect(self, station_id, station_value):
            """Connect to another station"""
            self.connections.append({"id": station_id, "value": station_value})
            
            # Log connection
            if self.loki:
                self.loki.log_event(
                    {"component": "station", "station": self.station_id, "action": "connect"},
                    f"Connected to station {station_id} with value {station_value}"
                )
        
        async def route_message(self, message):
            """Route a message using the LLM"""
            if self.loki:
                self.loki.log_event(
                    {"component": "station", "station": self.station_id, "action": "route"},
                    f"Routing message: {json.dumps(message)}"
                )
            
            # Get routing history
            routing_history = await self._get_routing_history(message)
            
            # Get routing decision from LLM
            try:
                next_station = await self.llm_manager.get_routing_decision(
                    self.station_id,
                    message,
                    self.connections,
                    routing_history
                )
                
                return {
                    "action": "route",
                    "from": self.station_id,
                    "to": next_station,
                    "message": message
                }
            except Exception as e:
                # Fallback to numerical proximity
                destination_value = message.get("destination_value", 0)
                closest_station = None
                min_distance = float('infinity')
                
                for station in self.connections:
                    distance = abs(station["value"] - destination_value)
                    if distance < min_distance:
                        min_distance = distance
                        closest_station = station["id"]
                
                return {
                    "action": "route",
                    "from": self.station_id,
                    "to": closest_station,
                    "message": message,
                    "fallback": True
                }
        
        async def _get_routing_history(self, message):
            """Get routing history for a message"""
            if not self.loki:
                return []
            
            # Extract message ID
            message_id = message.get("id", "unknown")
            
            # Query Loki for routing history
            query = f'{{component="station", action="route"}} |= "{message_id}"'
            results = self.loki.query_logs(query, limit=10)
            
            history = []
            for stream in results.get("data", {}).get("result", []):
                for _, log_line in stream.get("values", []):
                    try:
                        if "Routing message" in log_line:
                            parts = log_line.split("Routing message")
                            if len(parts) > 1:
                                routing_data = json.loads(parts[1].strip())
                                history.append({
                                    "from": routing_data.get("from"),
                                    "to": routing_data.get("to")
                                })
                    except:
                        continue
            
            return history
        
        async def process_incoming(self, message):
            """Process an incoming message"""
            # Log message receipt
            if self.loki:
                self.loki.log_event(
                    {"component": "station", "station": self.station_id, "action": "receive"},
                    f"Received message: {json.dumps(message)}"
                )
            
            # Check if this is the final destination
            if message.get("destination_value") == self.station_value:
                # Log delivery
                if self.loki:
                    self.loki.log_event(
                        {"component": "station", "station": self.station_id, "action": "deliver"},
                        f"Message delivered: {json.dumps(message)}"
                    )
                
                # Process the message content with LLM if needed
                if "content" in message and isinstance(message["content"], str) and self.llm_manager:
                    response = await self.llm_manager.generate(
                        f"Process this message as station {self.station_id}: {message['content']}",
                        self.model_name,
                        max_tokens=100
                    )
                    
                    processed_content = response.get("text", "Message processed")
                else:
                    processed_content = "Message processed"
                
                return {
                    "status": "delivered",
                    "station": self.station_id,
                    "response": processed_content
                }
            
            # Otherwise, route to next station
            return await self.route_message(message)
        
        async def get_status(self):
            """Get station status"""
            return {
                "station_id": self.station_id,
                "value": self.station_value,
                "model": self.model_name,
                "connections": len(self.connections),
                "connected_to": [conn["id"] for conn in self.connections]
            }
    
    # Define the Horn Network Manager
    class HornNetworkManager:
        def __init__(self, llm_manager=None, loki=None):
            self.stations = {}  # station_id -> LLMStation
            self.horns = []  # List of horn IDs
            self.pods = []  # List of pod IDs
            self.llm_manager = llm_manager
            self.loki = loki
            
            # Log network creation
            if self.loki:
                self.loki.log_event(
                    {"component": "network_manager", "action": "create"},
                    "Gabriel's Horn Network Manager initialized"
                )
        
        def add_station(self, station_id, station_value, station_type="pod", model="gemma-2b"):
            """Add a station to the network"""
            station = LLMStation(station_id, station_value, model, self.llm_manager, self.loki)
            self.stations[station_id] = station
            
            if station_type == "horn":
                self.horns.append(station_id)
            else:
                self.pods.append(station_id)
            
            if self.loki:
                self.loki.log_event(
                    {"component": "network_manager", "action": "add_station", "station_type": station_type},
                    f"Added {station_type} station {station_id} with value {station_value} using model {model}"
                )
            
            return station
        
        def connect_stations(self, station1_id, station2_id):
            """Connect two stations"""
            if station1_id not in self.stations or station2_id not in self.stations:
                return False
            
            station1 = self.stations[station1_id]
            station2 = self.stations[station2_id]
            
            station1.connect(station2_id, station2.station_value)
            station2.connect(station1_id, station1.station_value)
            
            if self.loki:
                self.loki.log_event(
                    {"component": "network_manager", "action": "connect"},
                    f"Connected stations {station1_id} and {station2_id}"
                )
            
            return True
        
        def create_ring(self, station_ids, ring_id=None):
            """Connect stations in a ring topology"""
            if len(station_ids) < 2:
                return None
            
            # Generate ring ID if not provided
            if not ring_id:
                ring_id = f"ring-{uuid.uuid4().hex[:8]}"
            
            # Connect stations in a ring
            for i in range(len(station_ids)):
                next_idx = (i + 1) % len(station_ids)
                self.connect_stations(station_ids[i], station_ids[next_idx])
            
            if self.loki:
                self.loki.log_event(
                    {"component": "network_manager", "action": "create_ring", "ring_id": ring_id},
                    f"Created ring {ring_id} with stations: {station_ids}"
                )
            
            return ring_id
        
        async def send_message(self, source_id, destination_value, content, priority="normal"):
            """Send a message from source to a station with destination value"""
            if source_id not in self.stations:
                return {"error": "Source station not found"}
            
            # Generate message ID
            message_id = f"msg-{uuid.uuid4().hex}"
            
            # Create message
            message = {
                "id": message_id,
                "content": content,
                "source_id": source_id,
                "source_value": self.stations[source_id].station_value,
                "destination_value": destination_value,
                "priority": priority,
                "timestamp": time.time(),
                "hops": 0,
                "path": [source_id]
            }
            
            if self.loki:
                self.loki.log_event(
                    {"component": "network_manager", "action": "send", "message_id": message_id},
                    f"Sending message from {source_id} to value {destination_value}"
                )
            
            # Start routing from source
            return await self._route_message(message, source_id)
        
        async def _route_message(self, message, current_station_id, max_hops=10):
            """Route a message through the network"""
            if message["hops"] >= max_hops:
                if self.loki:
                    self.loki.log_event(
                        {"component": "network_manager", "action": "max_hops"},
                        f"Message exceeded max hops: {json.dumps(message)}"
                    )
                return {"error": "Max hops exceeded", "path": message["path"]}
            
            # Process at current station
            station = self.stations[current_station_id]
            result = await station.process_incoming(message)
            
            # Check if delivered
            if result.get("status") == "delivered":
                if self.loki:
                    self.loki.log_event(
                        {"component": "network_manager", "action": "delivered"},
                        f"Message delivered to {current_station_id}"
                    )
                return {
                    "status": "delivered",
                    "destination": current_station_id,
                    "path": message["path"],
                    "hops": message["hops"],
                    "response": result.get("response")
                }
            
            # Continue routing
            next_station_id = result.get("to")
            if not next_station_id or next_station_id not in self.stations:
                if self.loki:
                    self.loki.log_event(
                        {"component": "network_manager", "action": "no_route"},
                        f"No route from {current_station_id}"
                    )
                return {"error": "No route available", "path": message["path"]}
            
            # Update message for next hop
            message["hops"] += 1
            message["path"].append(next_station_id)
            
            # Route to next station
            return await self._route_message(message, next_station_id, max_hops)
        
        async def get_network_status(self):
            """Get status of the entire network"""
            # Get status of each station
            station_statuses = {}
            for station_id, station in self.stations.items():
                station_statuses[station_id] = await station.get_status()
            
            return {
                "stations": len(self.stations),
                "horns": len(self.horns),
                "pods": len(self.pods),
                "station_statuses": station_statuses
            }
    
    # Create FastAPI app
    app = FastAPI(title="CogniKube Enhanced", version="2.0.0")
    
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Create Loki observer
    loki = LokiObserver(os.environ.get("LOKI_URL", "http://localhost:3100"))
    
    # Create network manager
    network = HornNetworkManager(llm_manager, loki)
    
    @app.on_event("startup")
    async def startup_event():
        # Initialize the network
        horns = [
            ("horn1", 100, "gemma-2b"),
            ("horn2", 200, "hermes-2-pro-llama-3-7b"),
            ("horn3", 300, "qwen2.5-14b"),
            ("horn4", 400, "gemma-2b"),
            ("horn5", 500, "hermes-2-pro-llama-3-7b"),
            ("horn6", 600, "qwen2.5-14b"),
            ("horn7", 700, "gemma-2b")
        ]
        
        pods = [
            ("pod1", 150, "gemma-2b"),
            ("pod2", 250, "hermes-2-pro-llama-3-7b"),
            ("pod3", 350, "qwen2.5-14b"),
            ("pod4", 450, "gemma-2b"),
            ("pod5", 550, "hermes-2-pro-llama-3-7b"),
            ("pod6", 650, "qwen2.5-14b"),
            ("pod7", 750, "gemma-2b")
        ]
        
        # Add stations
        for horn_id, value, model in horns:
            network.add_station(horn_id, value, "horn", model)
        
        for pod_id, value, model in pods:
            network.add_station(pod_id, value, "pod", model)
        
        # Create horn ring
        horn_ids = [horn[0] for horn in horns]
        network.create_ring(horn_ids)
        
        # Connect pods to horns
        for i, (pod_id, _, _) in enumerate(pods):
            horn_id = horns[i][0]
            network.connect_stations(pod_id, horn_id)
        
        loki.log_event(
            {"component": "cognikube", "action": "startup"},
            "CogniKube Enhanced initialized with Gabriel's Horn network"
        )
    
    @app.get("/")
    async def root():
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CogniKube Enhanced</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff; }
                .container { max-width: 800px; margin: 0 auto; }
                h1 { color: #00f5d4; }
                .card { background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; margin-bottom: 20px; }
                .btn { background: #00f5d4; color: #1a1a2e; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                pre { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CogniKube Enhanced</h1>
                <div class="card">
                    <h2>Gabriel's Horn Network</h2>
                    <p>Enhanced network with real LLM integration and Loki monitoring</p>
                    <button class="btn" onclick="getStatus()">Get Network Status</button>
                </div>
                <div class="card">
                    <h2>Send Message</h2>
                    <p>
                        Source: <select id="source">
                            <option value="pod1">pod1 (150)</option>
                            <option value="pod2">pod2 (250)</option>
                            <option value="pod3">pod3 (350)</option>
                            <option value="pod4">pod4 (450)</option>
                            <option value="pod5">pod5 (550)</option>
                            <option value="pod6">pod6 (650)</option>
                            <option value="pod7">pod7 (750)</option>
                        </select>
                        Destination Value: <input type="number" id="destination" value="350" />
                    </p>
                    <p>
                        Message: <input type="text" id="message" value="Test message" style="width: 300px;" />
                        Priority: <select id="priority">
                            <option value="low">Low</option>
                            <option value="normal" selected>Normal</option>
                            <option value="high">High</option>
                            <option value="critical">Critical</option>
                        </select>
                    </p>
                    <button class="btn" onclick="sendMessage()">Send Message</button>
                </div>
                <div class="card">
                    <h2>Results</h2>
                    <pre id="results">No results yet</pre>
                </div>
            </div>
            
            <script>
                async function getStatus() {
                    const response = await fetch('/status');
                    const data = await response.json();
                    document.getElementById('results').textContent = JSON.stringify(data, null, 2);
                }
                
                async function sendMessage() {
                    const source = document.getElementById('source').value;
                    const destination = document.getElementById('destination').value;
                    const message = document.getElementById('message').value;
                    const priority = document.getElementById('priority').value;
                    
                    const response = await fetch('/send', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            source,
                            destination_value: parseInt(destination),
                            content: message,
                            priority
                        })
                    });
                    
                    const data = await response.json();
                    document.getElementById('results').textContent = JSON.stringify(data, null, 2);
                }
                
                // Initialize
                getStatus();
            </script>
        </body>
        </html>
        """)
    
    @app.get("/status")
    async def status():
        return await network.get_network_status()
    
    @app.post("/send")
    async def send_message(data: dict):
        source = data.get("source")
        destination_value = data.get("destination_value")
        content = data.get("content")
        priority = data.get("priority", "normal")
        
        if not source or not destination_value or not content:
            return {"error": "Missing required fields"}
        
        result = await network.send_message(source, destination_value, content, priority)
        return result
    
    @app.get("/logs")
    async def get_logs(query: str = '{component="network_manager"}', limit: int = 100):
        results = loki.query_logs(query, limit=limit)
        return results
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to CogniKube Enhanced"
        })
        
        try:
            while True:
                data = await websocket.receive_json()
                
                if data.get("action") == "send":
                    result = await network.send_message(
                        data.get("source"),
                        data.get("destination_value"),
                        data.get("content"),
                        data.get("priority", "normal")
                    )
                    await websocket.send_json(result)
                
                elif data.get("action") == "status":
                    status = await network.get_network_status()
                    await websocket.send_json(status)
                
                else:
                    await websocket.send_json({"error": "Unknown action"})
        
        except Exception as e:
            if loki:
                loki.log_event(
                    {"component": "cognikube", "action": "websocket_error"},
                    f"WebSocket error: {str(e)}",
                    level="error"
                )
    
    return app

if __name__ == "__main__":
    # Deploy to Modal
    modal.run(app)