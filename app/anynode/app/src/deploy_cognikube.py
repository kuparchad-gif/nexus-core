#!/usr/bin/env python3
"""
Deploy CogniKube to Modal
"""
import modal
import os
import sys

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
    "numpy==1.26.1",
    "sentence-transformers==2.2.2",
    "qdrant-client==1.7.0",
    "transformers==4.35.0"
])

# Create volume for persistence
volume = modal.Volume.from_name("cognikube-brain", create_if_missing=True)

# Include local modules
with open("horn_network_manager.py", "r") as f:
    horn_network_manager_code = f.read()

with open("llm_station.py", "r") as f:
    llm_station_code = f.read()

with open("loki_integration.py", "r") as f:
    loki_integration_code = f.read()

@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/brain": volume}
)
@modal.asgi_app()
def cognikube_app():
    """Main CogniKube application"""
    # Write module files
    with open("horn_network_manager.py", "w") as f:
        f.write(horn_network_manager_code)
    
    with open("llm_station.py", "w") as f:
        f.write(llm_station_code)
    
    with open("loki_integration.py", "w") as f:
        f.write(loki_integration_code)
    
    # Now import the modules
    from fastapi import FastAPI, WebSocket
    from horn_network_manager import HornNetworkManager
    from loki_integration import loki
    import asyncio
    import json
    
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