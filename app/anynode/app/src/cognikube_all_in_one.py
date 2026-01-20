#!/usr/bin/env python3
"""
CogniKube All-in-One Deployment
"""
import modal
import asyncio
import json
import time
import requests
from typing import Dict, Any, List

# Create Modal app
app = modal.App("cognikube-master")

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
    "numpy==1.26.1"
])

# Create volume for persistence
volume = modal.Volume.from_name("cognikube-brain", create_if_missing=True)

@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/brain": volume}
)
@modal.asgi_app()
def cognikube_app():
    """Main CogniKube application with all components included"""
    from fastapi import FastAPI, WebSocket
    import asyncio
    import json
    import time
    import requests
    from typing import Dict, Any, List
    
    # ===== LOKI INTEGRATION =====
    class LokiLogger:
        """Loki logging integration"""
        
        def __init__(self, loki_url="http://localhost:3100"):
            self.loki_url = loki_url
            self.endpoint = f"{loki_url}/loki/api/v1/push"
        
        def log_event(self, labels: Dict[str, str], message: str):
            """Send log to Loki"""
            timestamp = str(int(time.time() * 1000000000))  # nanoseconds
            
            payload = {
                "streams": [
                    {
                        "stream": labels,
                        "values": [[timestamp, message]]
                    }
                ]
            }
            
            try:
                requests.post(self.endpoint, json=payload)
            except Exception as e:
                print(f"Loki logging error: {e}")
        
        def query_logs(self, query_string: str, limit: int = 100):
            """Query Loki logs"""
            query_endpoint = f"{self.loki_url}/loki/api/v1/query_range"
            
            params = {
                "query": query_string,
                "limit": limit,
                "start": int(time.time() - 3600) * 1000000000,  # 1 hour ago
                "end": int(time.time()) * 1000000000
            }
            
            try:
                response = requests.get(query_endpoint, params=params)
                data = response.json()
                return data
            except Exception as e:
                print(f"Loki query error: {e}")
                return {"data": {"result": []}}
    
    # Create Loki logger
    loki = LokiLogger()
    
    # ===== LLM STATION =====
    class LLMStation:
        def __init__(self, station_id: str, station_value: int, model_name: str = "gemma-2b"):
            self.station_id = station_id
            self.station_value = station_value
            self.model_name = model_name
            self.connections = []
            self.system_prompt = f"""You are station {station_id} with value {station_value}.
    Your job is to route messages to their destination efficiently."""
        
        def connect(self, station_id: str, station_value: int):
            """Connect to another station"""
            self.connections.append({"id": station_id, "value": station_value})
            loki.log_event(
                {"station": self.station_id, "action": "connect"},
                f"Connected to station {station_id} with value {station_value}"
            )
        
        async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
            """Route a message using the LLM"""
            loki.log_event(
                {"station": self.station_id, "action": "route"},
                f"Routing message: {json.dumps(message)}"
            )
            
            # Get routing decision based on numerical proximity
            next_station = self._get_routing_decision(message)
            
            return {
                "action": "route",
                "from": self.station_id,
                "to": next_station,
                "message": message
            }
        
        def _get_routing_decision(self, message: Dict[str, Any]) -> str:
            """Get routing decision based on numerical proximity"""
            if not self.connections:
                return None
            
            destination_value = message.get("destination_value", 0)
            
            # Find closest station by value
            closest_station = None
            min_distance = float('infinity')
            
            for station in self.connections:
                distance = abs(station["value"] - destination_value)
                if distance < min_distance:
                    min_distance = distance
                    closest_station = station["id"]
            
            return closest_station
        
        async def process_incoming(self, message: Dict[str, Any]) -> Dict[str, Any]:
            """Process an incoming message"""
            # Check if this is the final destination
            if message.get("destination_value") == self.station_value:
                loki.log_event(
                    {"station": self.station_id, "action": "deliver"},
                    f"Message delivered: {json.dumps(message)}"
                )
                return {"status": "delivered", "station": self.station_id}
            
            # Otherwise, route to next station
            return await self.route_message(message)
        
        async def get_status(self) -> Dict[str, Any]:
            """Get station status"""
            return {
                "station_id": self.station_id,
                "value": self.station_value,
                "model": self.model_name,
                "connections": len(self.connections)
            }
    
    # ===== HORN NETWORK MANAGER =====
    class HornNetworkManager:
        def __init__(self):
            self.stations = {}  # station_id -> LLMStation
            self.horns = []  # List of horn IDs (special routing stations)
            self.pods = []  # List of pod IDs (endpoint stations)
        
        def add_station(self, station_id: str, station_value: int, station_type: str = "pod", model: str = "gemma-2b"):
            """Add a station to the network"""
            station = LLMStation(station_id, station_value, model)
            self.stations[station_id] = station
            
            if station_type == "horn":
                self.horns.append(station_id)
            else:
                self.pods.append(station_id)
            
            loki.log_event(
                {"component": "network_manager", "action": "add_station"},
                f"Added {station_type} station {station_id} with value {station_value}"
            )
            
            return station
        
        def connect_stations(self, station1_id: str, station2_id: str):
            """Connect two stations"""
            if station1_id not in self.stations or station2_id not in self.stations:
                return False
            
            station1 = self.stations[station1_id]
            station2 = self.stations[station2_id]
            
            station1.connect(station2_id, station2.station_value)
            station2.connect(station1_id, station1.station_value)
            
            loki.log_event(
                {"component": "network_manager", "action": "connect"},
                f"Connected stations {station1_id} and {station2_id}"
            )
            
            return True
        
        def create_ring(self, station_ids: List[str]):
            """Connect stations in a ring topology"""
            if len(station_ids) < 2:
                return False
            
            for i in range(len(station_ids)):
                next_idx = (i + 1) % len(station_ids)
                self.connect_stations(station_ids[i], station_ids[next_idx])
            
            loki.log_event(
                {"component": "network_manager", "action": "create_ring"},
                f"Created ring with stations: {station_ids}"
            )
            
            return True
        
        async def send_message(self, source_id: str, destination_value: int, content: Any, priority: str = "normal") -> Dict[str, Any]:
            """Send a message from source to a station with destination value"""
            if source_id not in self.stations:
                return {"error": "Source station not found"}
            
            message = {
                "content": content,
                "source_id": source_id,
                "source_value": self.stations[source_id].station_value,
                "destination_value": destination_value,
                "priority": priority,
                "timestamp": time.time(),
                "hops": 0,
                "path": [source_id]
            }
            
            loki.log_event(
                {"component": "network_manager", "action": "send"},
                f"Sending message from {source_id} to value {destination_value}"
            )
            
            # Start routing from source
            return await self._route_message(message, source_id)
        
        async def _route_message(self, message: Dict[str, Any], current_station_id: str, max_hops: int = 10) -> Dict[str, Any]:
            """Route a message through the network"""
            if message["hops"] >= max_hops:
                loki.log_event(
                    {"component": "network_manager", "action": "max_hops"},
                    f"Message exceeded max hops: {json.dumps(message)}"
                )
                return {"error": "Max hops exceeded", "path": message["path"]}
            
            # Process at current station
            station = self.stations[current_station_id]
            result = await station.process_incoming(message)
            
            # Check if delivered
            if result.get("status") == "delivered":
                loki.log_event(
                    {"component": "network_manager", "action": "delivered"},
                    f"Message delivered to {current_station_id}"
                )
                return {
                    "status": "delivered",
                    "destination": current_station_id,
                    "path": message["path"],
                    "hops": message["hops"]
                }
            
            # Continue routing
            next_station_id = result.get("to")
            if not next_station_id or next_station_id not in self.stations:
                loki.log_event(
                    {"component": "network_manager", "action": "no_route"},
                    f"No route from {current_station_id}"
                )
                return {"error": "No route available", "path": message["path"]}
            
            # Update message for next hop
            message["hops"] += 1
            message["path"].append(next_station_id)
            
            # Route to next station
            return await self._route_message(message, next_station_id, max_hops)
        
        async def get_network_status(self) -> Dict[str, Any]:
            """Get status of the entire network"""
            station_statuses = {}
            
            for station_id, station in self.stations.items():
                station_statuses[station_id] = await station.get_status()
            
            return {
                "stations": len(self.stations),
                "horns": len(self.horns),
                "pods": len(self.pods),
                "station_statuses": station_statuses
            }
    
    # ===== FASTAPI APPLICATION =====
    app = FastAPI(title="CogniKube Master", version="1.0.0")
    
    # Initialize network
    network = HornNetworkManager()
    
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
            "CogniKube initialized with Gabriel's Horn network"
        )
    
    @app.get("/")
    async def root():
        return {"status": "active", "name": "CogniKube Master"}
    
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
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to CogniKube"
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
            loki.log_event(
                {"component": "cognikube", "action": "websocket_error"},
                f"WebSocket error: {str(e)}"
            )
    
    return app

if __name__ == "__main__":
    # Deploy to Modal
    modal.run(app)