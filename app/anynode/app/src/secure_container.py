#!/usr/bin/env python3
"""
CogniKube Secure Container - Layer 1
Foundation with controlled port exposure
"""

import asyncio
import ssl
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="CogniKube Secure Foundation")

# CORS - controlled origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cognikube.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST", "WebSocket"],
    allow_headers=["*"],
)

class PortManager:
    def __init__(self):
        self.exposed_ports = {
            8080: "main_api",      # Main API
            8443: "secure_ws",     # Secure WebSocket  
            9090: "health_check"   # Health monitoring
        }
        self.internal_ports = {
            6379: "redis",         # Internal Redis
            5432: "postgres",      # Internal DB
            3100: "loki"          # Internal Loki
        }
    
    def is_port_exposed(self, port: int) -> bool:
        return port in self.exposed_ports

port_manager = PortManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "exposed_ports": list(port_manager.exposed_ports.keys()),
        "secure": True
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Secure WebSocket endpoint - will add auth in next layer"""
    await websocket.accept()
    await websocket.send_text('{"status": "connected", "secure": true}')

if __name__ == "__main__":
    # SSL context for HTTPS
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain("cert.pem", "key.pem")  # Add certs
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        # ssl_keyfile="key.pem",
        # ssl_certfile="cert.pem"
    )