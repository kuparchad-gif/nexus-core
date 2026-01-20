#!/usr/bin/env python3
"""
CogniKube Load Balancer - Multi-endpoint resilience
"""

import asyncio
import aiohttp
from fastapi import FastAPI, WebSocket
import random
import httpx

app = FastAPI(title="CogniKube Load Balancer")

class EndpointManager:
    def __init__(self):
        self.endpoints = [
            "https://cognikube-1.modal.run",
            "https://cognikube-2.modal.run", 
            "https://cognikube-3.modal.run"
        ]
        self.healthy_endpoints = []
        self.discovery_urls = [
            "https://aethereal-nexus-viren-db0--cognikube-complete-cognikube--4f4b9b.modal.run",
            "https://aethereal-nexus-viren-db1--cognikube-networked-cognikube--platform.modal.run"
        ]
        self.endpoint_stats = {}  # Track performance metrics
        self.discovery_registry = "https://registry.cognikube.io/endpoints"  # Central registry
        
    async def health_check(self):
        """Check endpoint health and discover new endpoints"""
        healthy = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check static endpoints
            for endpoint in self.endpoints:
                try:
                    start_time = asyncio.get_event_loop().time()
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{endpoint}/health", timeout=5) as resp:
                            response_time = asyncio.get_event_loop().time() - start_time
                            if resp.status == 200:
                                healthy.append(endpoint)
                                data = await resp.json()
                                # Store performance metrics
                                self.endpoint_stats[endpoint] = {
                                    "response_time": response_time,
                                    "load": data.get("load", 0.5),
                                    "memory": data.get("memory", 0.5),
                                    "last_check": asyncio.get_event_loop().time()
                                }
                except Exception as e:
                    print(f"Health check failed for {endpoint}: {str(e)}")
            
            # Discover new endpoints from discovery URLs
            for url in self.discovery_urls:
                try:
                    response = await client.get(f"{url}/health")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "healthy" and url not in self.endpoints:
                            self.endpoints.append(url)
                            healthy.append(url)
                except Exception as e:
                    print(f"Discovery failed for {url}: {str(e)}")
            
            # Check central registry for new endpoints
            try:
                registry_response = await client.get(self.discovery_registry)
                if registry_response.status_code == 200:
                    registry_data = registry_response.json()
                    for endpoint_data in registry_data.get("endpoints", []):
                        endpoint_url = endpoint_data.get("url")
                        if endpoint_url and endpoint_url not in self.endpoints:
                            # Verify new endpoint
                            try:
                                verify_response = await client.get(f"{endpoint_url}/health", timeout=3)
                                if verify_response.status_code == 200:
                                    self.endpoints.append(endpoint_url)
                                    healthy.append(endpoint_url)
                                    print(f"Discovered new endpoint: {endpoint_url}")
                            except:
                                pass
            except Exception as e:
                print(f"Registry check failed: {str(e)}")
        
        self.healthy_endpoints = healthy
        
        # Report our status to the registry
        try:
            await self._report_to_registry()
        except Exception as e:
            print(f"Failed to report to registry: {str(e)}")
    
    async def _report_to_registry(self):
        """Report this load balancer's status to the central registry"""
        import socket
        hostname = socket.gethostname()
        
        report_data = {
            "hostname": hostname,
            "healthy_endpoints": len(self.healthy_endpoints),
            "total_endpoints": len(self.endpoints),
            "timestamp": datetime.now().isoformat()
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.discovery_registry}/report",
                json=report_data,
                timeout=5.0
            )
        
    def get_endpoint(self):
        """Get healthy endpoint with load balancing"""
        if not self.healthy_endpoints:
            return None
        return random.choice(self.healthy_endpoints)

manager = EndpointManager()

@app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    
    endpoint = manager.get_endpoint()
    if not endpoint:
        await websocket.send_text('{"error": "No healthy endpoints"}')
        return
        
    ws_url = endpoint.replace("https://", "wss://") + "/ws"
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as backend_ws:
            async def forward_to_backend():
                async for msg in websocket.iter_text():
                    await backend_ws.send_str(msg)
                    
            async def forward_to_client():
                async for msg in backend_ws:
                    await websocket.send_text(msg.data)
                    
            await asyncio.gather(
                forward_to_backend(),
                forward_to_client()
            )

@app.on_event("startup")
async def startup():
    asyncio.create_task(periodic_health_check())
    
async def periodic_health_check():
    while True:
        await manager.health_check()
        await asyncio.sleep(30)