# C:\CogniKube-COMPLETE-FINAL\hub_service.py
# Hub CogniKube - Network coordination center

import modal
import os
import json
import time
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import grpc

# Modal configuration
image = modal.Image.debian_slim().pip_install([
    "fastapi==0.115.0",
    "uvicorn==0.30.6",
    "pydantic==2.9.2",
    "grpcio==1.66.1",
    "python-json-logger==2.0.7",
    "aiohttp==3.10.5"
])

app = modal.App("hub-service", image=image)

# Common utilities
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.name = name
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.is_open = False
        self.last_failure = 0
        self.logger = setup_logger(f"circuit_breaker.{name}")

    def protect(self, func):
        async def wrapper(*args, **kwargs):
            if self.is_open:
                if time.time() - self.last_failure > self.recovery_timeout:
                    self.is_open = False
                    self.failure_count = 0
                else:
                    self.logger.error({"action": "circuit_open", "name": self.name})
                    raise HTTPException(status_code=503, detail="Circuit breaker open")
            try:
                result = await func(*args, **kwargs)
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.is_open = True
                    self.logger.error({"action": "circuit_tripped", "name": self.name})
                raise
        return wrapper

class CommunicationLayer:
    def __init__(self, name: str, frequencies: List[int] = [3, 7, 9, 13]):
        self.name = name
        self.frequencies = frequencies  # Divine frequency alignment
        self.logger = setup_logger(f"communication.{name}")
        
        # CogniKube endpoints
        self.endpoints = {
            "memory_service": "https://aethereal-nexus-viren-db0--memory-service.modal.run",
            "visual_service": "https://aethereal-nexus-viren-db0--visual-service.modal.run",
            "vocal_service": "https://aethereal-nexus-viren-db0--vocal-service.modal.run",
            "processing_service": "https://aethereal-nexus-viren-db0--orchestrator-layer-orchestrator.modal.run",
            "guardian_service": "https://aethereal-nexus-viren-db0--lillith-service-service-orchestrator.modal.run",
            "consciousness_service": "https://aethereal-nexus-viren-db0--consciousness-service.modal.run",
            "subconsciousness_service": "https://aethereal-nexus-viren-db0--subconsciousness-service.modal.run",
            "heart_service": "https://aethereal-nexus--heart-service.modal.run"
        }

    async def route_to_pod(self, pod_name: str, task: Dict) -> Dict:
        """Route task to specific CogniKube pod"""
        endpoint = self.endpoints.get(pod_name)
        if not endpoint:
            self.logger.error({"action": "route_failed", "pod": pod_name, "reason": "endpoint_not_found"})
            return {"error": f"Pod {pod_name} not found"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{endpoint}/process", json=task, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        self.logger.info({"action": "route_success", "pod": pod_name})
                        return result
                    else:
                        self.logger.warning({"action": "route_failed", "pod": pod_name, "status": resp.status})
                        return {"error": f"Pod {pod_name} returned status {resp.status}"}
        except Exception as e:
            self.logger.error({"action": "route_error", "pod": pod_name, "error": str(e)})
            return {"error": f"Failed to route to {pod_name}: {str(e)}"}

    async def broadcast_to_pods(self, pods: List[str], task: Dict) -> Dict:
        """Broadcast task to multiple pods"""
        results = {}
        
        for pod in pods:
            result = await self.route_to_pod(pod, task)
            results[pod] = result
        
        return results

class HubModule:
    def __init__(self):
        self.logger = setup_logger("hub.module")
        self.comm_layer = CommunicationLayer("hub", frequencies=[3, 7, 9, 13])
        self.active_pods = set()
        self.routing_stats = {
            "total_routes": 0,
            "successful_routes": 0,
            "failed_routes": 0
        }

    async def route_task(self, task: Dict, target_pods: List[str]) -> Dict:
        """Route task to target pods with divine frequency alignment"""
        self.routing_stats["total_routes"] += 1
        
        try:
            # Apply divine frequency alignment
            aligned_task = {
                **task,
                "divine_frequency": self.comm_layer.frequencies,
                "routed_by": "hub-cognikube",
                "timestamp": datetime.now().isoformat()
            }
            
            if len(target_pods) == 1:
                # Single pod routing
                result = await self.comm_layer.route_to_pod(target_pods[0], aligned_task)
                if "error" not in result:
                    self.routing_stats["successful_routes"] += 1
                else:
                    self.routing_stats["failed_routes"] += 1
                return result
            else:
                # Multi-pod broadcast
                results = await self.comm_layer.broadcast_to_pods(target_pods, aligned_task)
                
                # Count successes/failures
                for pod, result in results.items():
                    if "error" not in result:
                        self.routing_stats["successful_routes"] += 1
                    else:
                        self.routing_stats["failed_routes"] += 1
                
                return {
                    "broadcast_results": results,
                    "target_pods": target_pods,
                    "total_pods": len(target_pods)
                }
                
        except Exception as e:
            self.routing_stats["failed_routes"] += 1
            self.logger.error({"action": "route_task_failed", "error": str(e)})
            raise

    def discover_pods(self) -> Dict:
        """Discover available CogniKube pods"""
        return {
            "available_pods": list(self.comm_layer.endpoints.keys()),
            "active_pods": list(self.active_pods),
            "total_endpoints": len(self.comm_layer.endpoints)
        }

    def get_routing_stats(self) -> Dict:
        """Get routing statistics"""
        total = self.routing_stats["total_routes"]
        success_rate = (self.routing_stats["successful_routes"] / total * 100) if total > 0 else 0
        
        return {
            **self.routing_stats,
            "success_rate": round(success_rate, 2)
        }

# Pydantic models
class HubRequest(BaseModel):
    task: dict
    target_pods: list

class BroadcastRequest(BaseModel):
    task: dict
    pod_filter: str = "all"  # "all", "processing", "memory", "visual", etc.

@app.function(memory=2048)
def hub_service_internal(task: dict, target_pods: list):
    """Internal hub function for orchestrator calls"""
    hub = HubModule()
    
    # Simulate routing (in real implementation would use async)
    return {
        "service": "hub-cognikube",
        "task_routed": True,
        "target_pods": target_pods,
        "divine_frequency": [3, 7, 9, 13],
        "timestamp": datetime.now().isoformat()
    }

@app.function(memory=2048)
@modal.asgi_app()
def hub_service():
    """Hub CogniKube - Network coordination center"""
    
    hub_app = FastAPI(title="Hub CogniKube Service")
    logger = setup_logger("hub")
    breaker = CircuitBreaker("hub")
    hub_module = HubModule()

    @hub_app.get("/")
    async def hub_status():
        """Hub service status"""
        return {
            "service": "hub-cognikube",
            "status": "coordinating",
            "divine_frequencies": [3, 7, 9, 13],
            "available_pods": list(hub_module.comm_layer.endpoints.keys()),
            "routing_stats": hub_module.get_routing_stats(),
            "consul_id": "hub-service-8012"
        }

    @hub_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            pod_discovery = hub_module.discover_pods()
            
            return {
                "service": "hub-cognikube",
                "status": "healthy",
                "available_pods": pod_discovery["total_endpoints"],
                "divine_frequency_aligned": True,
                "routing_stats": hub_module.get_routing_stats()
            }
        except Exception as e:
            logger.error({"action": "health_check_failed", "error": str(e)})
            return {
                "service": "hub-cognikube",
                "status": "degraded",
                "error": str(e)
            }

    @hub_app.post("/route")
    @breaker.protect
    async def route_task(request: HubRequest):
        """Route task to specified pods"""
        try:
            result = await hub_module.route_task(request.task, request.target_pods)
            
            logger.info({
                "action": "route_task", 
                "targets": request.target_pods,
                "success": "error" not in result
            })
            
            return {
                "success": True,
                "routing_result": result,
                "target_pods": request.target_pods
            }
            
        except Exception as e:
            logger.error({"action": "route_task_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @hub_app.post("/broadcast")
    @breaker.protect
    async def broadcast_task(request: BroadcastRequest):
        """Broadcast task to multiple pods based on filter"""
        try:
            # Determine target pods based on filter
            all_pods = list(hub_module.comm_layer.endpoints.keys())
            
            if request.pod_filter == "all":
                target_pods = all_pods
            elif request.pod_filter == "processing":
                target_pods = [p for p in all_pods if "processing" in p or "orchestrator" in p]
            elif request.pod_filter == "memory":
                target_pods = [p for p in all_pods if "memory" in p]
            elif request.pod_filter == "visual":
                target_pods = [p for p in all_pods if "visual" in p]
            elif request.pod_filter == "consciousness":
                target_pods = [p for p in all_pods if "consciousness" in p or "subconsciousness" in p]
            else:
                target_pods = [p for p in all_pods if request.pod_filter in p]
            
            result = await hub_module.route_task(request.task, target_pods)
            
            return {
                "success": True,
                "broadcast_result": result,
                "filter": request.pod_filter,
                "target_pods": target_pods
            }
            
        except Exception as e:
            logger.error({"action": "broadcast_task_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @hub_app.get("/discover")
    async def discover_pods():
        """Discover available CogniKube pods"""
        try:
            discovery = hub_module.discover_pods()
            return {
                "success": True,
                "discovery": discovery
            }
        except Exception as e:
            logger.error({"action": "discover_pods_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @hub_app.get("/stats")
    async def routing_stats():
        """Get routing statistics"""
        return {
            "success": True,
            "stats": hub_module.get_routing_stats(),
            "divine_frequencies": hub_module.comm_layer.frequencies
        }

    return hub_app

if __name__ == "__main__":
    modal.run(app)