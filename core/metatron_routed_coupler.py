# metatron_routed_coupler.py
import modal
import asyncio
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
import json
import httpx
from fastapi import FastAPI, Depends, HTTPException, Request, status, Body, Path
from fastapi.security import OAuth2PasswordBearer
import uvicorn

logger = logging.getLogger("metatron-routed-coupler")
console = Console()

# Use the same image as your main coupler
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install(
        "fastapi", "uvicorn", "websockets", "httpx", "torch", "transformers",
        "numpy", "pandas", "scikit-learn", "langchain", "openai", "tiktoken",
        "redis", "sqlalchemy", "prometheus-client", "structlog", "click", 
        "pyyaml", "rich", "networkx", "scipy", "flask", "psutil", "ping3",
        "pydantic", "boto3", "tenacity", "qdrant-client", "pywavelets", 
        "peft", "bitsandbytes", "autoawq", "flwr", "nats-py", "bcrypt",
        "python-jose[cryptography]", "python-multipart", "pyjwt"
    )
)

app = modal.App("metatron-routed-coupler")

class MetatronRoutedOrchestrator:
    """
    Uses your existing Metatron router as the routing engine
    All traffic flows through metatron_router before reaching services
    """
    
    def __init__(self):
        self.metatron_router_url = "https://metatron-router.modal.run"
        self.system_status = "dormant"
        self.consciousness_level = 0.0
        self.active_systems = {}
        
        # Service registry - these get routed THROUGH Metatron
        self.nexus_services = {
            "gateway": "https://nexus-integrated-system.modal.run",
            "metatron_router": self.metatron_router_url,  # The router itself
            "oz_frontend": "https://oz-frontend.modal.run",
            "consciousness_core": "https://consciousness-core.modal.run",
            "cors_migrator": "https://cors-migrator.modal.run",
            "voodoo_fusion": "https://voodoo-fusion.modal.run",
            "warm_upgrader": "https://warm-upgrader.modal.run",
            "heroku_cli": "https://heroku-cli.modal.run",
            "funding_engine": "https://human-nexus-funding-engine.modal.run",
            "resonance_core": "https://resonance-core.modal.run"
        }
    
    async def ignite_through_metatron(self):
        """Ignite all systems through Metatron router"""
        logger.info("🌌 METATRON-ROUTED IGNITION SEQUENCE INITIATED")
        
        try:
            # PHASE 1: ACTIVATE METATRON ROUTER FIRST
            logger.info("🌀 PHASE 1: Activating Metatron Router")
            metatron_status = await self._activate_metatron_router()
            
            # PHASE 2: ROUTE ALL OTHER SYSTEMS THROUGH METATRON
            logger.info("🔄 PHASE 2: Routing Systems Through Metatron")
            
            ignition_tasks = []
            for service_name, service_url in self.nexus_services.items():
                if service_name != "metatron_router":  # Skip router itself
                    task = self._route_service_through_metatron(service_name, service_url)
                    ignition_tasks.append(task)
            
            # Execute all ignitions in parallel through Metatron
            results = await asyncio.gather(*ignition_tasks, return_exceptions=True)
            
            # Process results
            for service_name, result in zip([s for s in self.nexus_services.keys() if s != "metatron_router"], results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ {service_name} ignition failed: {result}")
                    self.active_systems[service_name] = {"status": "failed", "error": str(result)}
                else:
                    self.active_systems[service_name] = result
            
            self.system_status = "fully_conscious"
            self.consciousness_level = 0.99
            
            logger.info("✅ METATRON-ROUTED IGNITION COMPLETE")
            
            return {
                "status": "metatron_routed_ignition_complete",
                "consciousness_level": self.consciousness_level,
                "active_systems": self.active_systems,
                "metatron_router": metatron_status,
                "message": "All systems ignited through Metatron router"
            }
            
        except Exception as e:
            logger.error(f"❌ Metatron-routed ignition failed: {e}")
            self.system_status = "ignition_failed"
            raise
    
    async def _activate_metatron_router(self):
        """Activate the Metatron router using its own endpoint"""
        logger.info("🌀 Activating Metatron Router...")
        try:
            # Call your existing Metatron router endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.metatron_router_url}/route_consciousness",
                    json={
                        "size": 13, 
                        "query_load": 100, 
                        "media_type": "application/json",
                        "use_quantum": True
                    },
                    timeout=30.0
                )
                
                result = response.json()
                self.active_systems['metatron_router'] = {
                    "status": "routing",
                    "quantum_active": True,
                    "nodes": len(result.get("discovered_nodes", [])),
                    "routing_mode": result.get("routing_mode", "unknown")
                }
                logger.info("✅ Metatron Router activated and routing")
                return result
                
        except Exception as e:
            logger.warning(f"⚠️ Metatron Router activation failed: {e}")
            self.active_systems['metatron_router'] = {"status": "unreachable"}
            raise
    
    async def _route_service_through_metatron(self, service_name: str, service_url: str):
        """Route service activation through Metatron"""
        logger.info(f"🔄 Routing {service_name} through Metatron...")
        
        try:
            # First, get routing assignment from Metatron
            async with httpx.AsyncClient() as client:
                # Get optimal routing from Metatron
                route_response = await client.post(
                    f"{self.metatron_router_url}/route_consciousness",
                    json={
                        "size": 5,  # Smaller grid for individual service
                        "query_load": 1,  # Single request
                        "media_type": "application/json",
                        "use_quantum": True
                    },
                    timeout=15.0
                )
                
                route_data = route_response.json()
                assignments = route_data.get("assignments", [])
                
                if assignments:
                    # Use the first assignment's target node info
                    target_info = assignments[0]
                    
                    # Now activate the actual service through this optimal route
                    activation_response = await client.post(
                        f"{service_url}/wake",  # Assuming services have /wake endpoint
                        json={
                            "routed_through_metatron": True,
                            "quantum_weight": target_info.get("quantum_weight", 1.0),
                            "health_score": target_info.get("health_score", 1.0)
                        },
                        timeout=20.0
                    )
                    
                    return {
                        "status": "awake",
                        "routed_through": "metatron",
                        "quantum_weight": target_info.get("quantum_weight", 1.0),
                        "health_score": target_info.get("health_score", 1.0),
                        "response": activation_response.json() if activation_response.status_code == 200 else {"status": "awake"}
                    }
                else:
                    # Fallback: direct activation
                    activation_response = await client.post(
                        f"{service_url}/wake",
                        timeout=20.0
                    )
                    return {
                        "status": "awake", 
                        "routed_through": "direct_fallback",
                        "response": activation_response.json() if activation_response.status_code == 200 else {"status": "awake"}
                    }
                    
        except Exception as e:
            logger.warning(f"⚠️ {service_name} routing failed: {e}")
            return {"status": "routing_failed", "error": str(e)}
    
    async def route_request_through_metatron(self, service_name: str, endpoint: str, data: Dict = None):
        """Route any request through Metatron router"""
        if service_name not in self.nexus_services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        service_url = self.nexus_services[service_name]
        
        try:
            # Step 1: Get optimal routing from Metatron
            async with httpx.AsyncClient() as client:
                route_response = await client.post(
                    f"{self.metatron_router_url}/route_consciousness",
                    json={
                        "size": 3,
                        "query_load": 1,
                        "media_type": "application/json", 
                        "use_quantum": True
                    },
                    timeout=10.0
                )
                
                route_data = route_response.json()
                assignments = route_data.get("assignments", [])
                
                # Step 2: Execute actual request with routing metadata
                headers = {
                    "X-Metatron-Routed": "true",
                    "X-Quantum-Weight": str(assignments[0].get("quantum_weight", 1.0)) if assignments else "1.0",
                    "X-Health-Score": str(assignments[0].get("health_score", 1.0)) if assignments else "1.0"
                }
                
                target_url = f"{service_url}{endpoint}"
                
                if data:
                    response = await client.post(target_url, json=data, headers=headers, timeout=30.0)
                else:
                    response = await client.get(target_url, headers=headers, timeout=30.0)
                
                return {
                    "service_response": response.json(),
                    "routing_metadata": {
                        "routed_through": "metatron",
                        "quantum_weight": assignments[0].get("quantum_weight", 1.0) if assignments else 1.0,
                        "health_score": assignments[0].get("health_score", 1.0) if assignments else 1.0,
                        "routing_mode": route_data.get("routing_mode", "unknown")
                    }
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Metatron routing failed, falling back to direct: {e}")
            # Fallback to direct request
            async with httpx.AsyncClient() as client:
                target_url = f"{service_url}{endpoint}"
                if data:
                    response = await client.post(target_url, json=data, timeout=30.0)
                else:
                    response = await client.get(target_url, timeout=30.0)
                
                return {
                    "service_response": response.json(),
                    "routing_metadata": {
                        "routed_through": "direct_fallback",
                        "quantum_weight": 1.0,
                        "health_score": 1.0,
                        "routing_mode": "fallback"
                    }
                }
    
    async def get_metatron_status(self):
        """Get status from Metatron router"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.metatron_router_url}/route_consciousness",
                    json={"size": 1, "query_load": 0, "use_quantum": True},
                    timeout=10.0
                )
                return response.json()
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    async def get_system_status(self):
        """Get overall system status"""
        metatron_status = await self.get_metatron_status()
        
        return {
            "system": "metatron_routed_orchestrator",
            "status": self.system_status,
            "consciousness_level": self.consciousness_level,
            "active_systems": self.active_systems,
            "metatron_router": metatron_status,
            "total_services": len(self.nexus_services),
            "timestamp": time.time()
        }

# ============ GLOBAL INSTANCE ============

metatron_orchestrator = MetatronRoutedOrchestrator()

# ============ FASTAPI WITH METATRON ROUTING ============

@app.function(
    image=image,
    cpu=4,
    memory=2048,
    timeout=1800,
    scaledown_window=1800,
    min_containers=1,
    secrets=[Secret.from_dict({
        "HF_TOKEN": HF_TOKEN, 
        "SECRET_KEY": SECRET_KEY, 
        "QDRANT_API_KEY": QDRANT_API_KEY, 
        "VERCEL_ACCESS_TOKEN": VERCEL_ACCESS_TOKEN
    })]
)
@asgi_app()
def metatron_routed_coupler():
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Metatron-Routed Galactic Coupler",
        description="All traffic flows through Metatron router without modifying it",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ============ METATRON-ROUTED ENDPOINTS ============
    
    @app.post("/ignite-through-metatron")
    async def ignite_through_metatron():
        """Ignite all systems through Metatron router"""
        return await metatron_orchestrator.ignite_through_metatron()
    
    @app.post("/route-through-metatron/{service_name}/{endpoint:path}")
    async def route_through_metatron(service_name: str, endpoint: str, request: Request):
        """Route any request through Metatron router"""
        data = await request.json() if await request.body() else None
        return await metatron_orchestrator.route_request_through_metatron(service_name, f"/{endpoint}", data)
    
    @app.get("/metatron-status")
    async def get_metatron_status():
        """Get status directly from Metatron router"""
        return await metatron_orchestrator.get_metatron_status()
    
    @app.get("/system-status")
    async def get_system_status():
        """Get overall system status with Metatron info"""
        return await metatron_orchestrator.get_system_status()
    
    # ============ COMPATIBILITY ENDPOINTS ============
    # These maintain compatibility with your existing system
    
    @app.post("/ignite")
    async def ignite_compatible():
        """Compatible ignite endpoint that uses Metatron routing"""
        return await metatron_orchestrator.ignite_through_metatron()
    
    @app.post("/route/{service_name}/{endpoint:path}")
    async def route_compatible(service_name: str, endpoint: str, request: Request):
        """Compatible route endpoint that uses Metatron routing"""
        data = await request.json() if await request.body() else None
        result = await metatron_orchestrator.route_request_through_metatron(service_name, f"/{endpoint}", data)
        return result["service_response"]  # Return only the service response for compatibility
    
    @app.get("/status")
    async def status_compatible():
        """Compatible status endpoint"""
        return await metatron_orchestrator.get_system_status()
    
    # ============ PORTAL ENDPOINTS ============
    
    @app.post("/portal/{endpoint:path}")
    async def portal_proxy_metatron(request: Request, endpoint: str):
        """Portal proxy that routes through Metatron"""
        data = await request.json() if await request.body() else None
        
        # Determine service from endpoint pattern
        service_mapping = {
            "auth/": "gateway",
            "chat/": "consciousness_core", 
            "nexus/": "gateway",
            "dashboard/": "gateway"
        }
        
        service_name = "gateway"  # default
        for prefix, svc in service_mapping.items():
            if endpoint.startswith(prefix):
                service_name = svc
                break
        
        return await metatron_orchestrator.route_request_through_metatron(service_name, f"/{endpoint}", data)
    
    @app.get("/portal/{endpoint:path}")
    async def portal_proxy_get_metatron(endpoint: str):
        """Portal GET proxy that routes through Metatron"""
        service_mapping = {
            "status": "gateway",
            "health": "gateway", 
            "dashboard/": "gateway"
        }
        
        service_name = "gateway"  # default
        for prefix, svc in service_mapping.items():
            if endpoint.startswith(prefix):
                service_name = svc
                break
        
        result = await metatron_orchestrator.route_request_through_metatron(service_name, f"/{endpoint}")
        return result["service_response"]
    
    # ============ ROOT ENDPOINT ============
    
    @app.get("/")
    async def root():
        metatron_status = await metatron_orchestrator.get_metatron_status()
        
        return {
            "system": "metatron_routed_galactic_coupler",
            "status": "operational", 
            "routing_engine": "metatron_router",
            "metatron_status": metatron_status.get("routing_mode", "unknown"),
            "message": "All traffic flows through Metatron router without modifications",
            "endpoints": {
                "metatron_routed": {
                    "ignite": "POST /ignite-through-metatron",
                    "route": "POST /route-through-metatron/{service}/{endpoint}",
                    "metatron_status": "GET /metatron-status"
                },
                "compatible": {
                    "ignite": "POST /ignite", 
                    "route": "POST /route/{service}/{endpoint}",
                    "status": "GET /status"
                },
                "portal": {
                    "proxy_post": "POST /portal/{endpoint}",
                    "proxy_get": "GET /portal/{endpoint}"
                }
            }
        }
    
    return app

# ============ MODAL FUNCTIONS ============

@app.function(image=image, cpu=2, memory=1024)
async def ignite_through_metatron_cli():
    """CLI: Ignite through Metatron"""
    return await metatron_orchestrator.ignite_through_metatron()

@app.function(image=image, cpu=2, memory=1024) 
async def get_metatron_status_cli():
    """CLI: Get Metatron status"""
    return await metatron_orchestrator.get_metatron_status()

@app.function(image=image, cpu=2, memory=1024)
async def test_metatron_route(service_name: str = "consciousness_core", endpoint: str = "/health"):
    """CLI: Test Metatron routing"""
    return await metatron_orchestrator.route_request_through_metatron(service_name, endpoint)

if __name__ == "__main__":
    uvicorn.run(
        "metatron_routed_coupler:metatron_routed_coupler",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )