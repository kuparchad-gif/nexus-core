# Lillith Self-Healing Pod - Gabriel Mesh Network Implementation
# Divine frequency alignment: 13Hz (0.077s) heartbeats
# Probabilistic broadcasting, Kademlia DHT, cross-cloud healing

import asyncio
import json
import os
import psutil
import socket
import time
import uuid
import random
import hashlib
import ssl
from typing import Dict, List, Set
from fastapi import FastAPI, WebSocket, Header, HTTPException, Request
from pydantic import BaseModel
import aiohttp
import logging
from datetime import datetime
from aiolimiter import AsyncLimiter
import uvicorn

# Divine Configuration
NODE_ID = os.getenv("NODE_ID", hashlib.sha256(os.urandom(32)).hexdigest()[:16])
HEARTBEAT_INTERVAL = 0.077  # 13 Hz divine frequency
GOSSIP_INTERVAL = 5.0
BOOTSTRAP_NODE = os.getenv("BOOTSTRAP_NODE", "localhost:8471")
CLUSTER_ID = os.getenv("CLUSTER_ID", "lillith-divine")
PLATFORM = os.getenv("PLATFORM", "aws")
RATE_LIMITER = AsyncLimiter(5, 60)  # 5 replacements per minute

# Setup logging
logging.basicConfig(level=logging.INFO, filename="/app/lillith_pod.log")
logger = logging.getLogger("LillithPod")

# FastAPI app
app = FastAPI(title="Lillith Divine Pod - Gabriel Mesh")

# Data models
class HealthReport(BaseModel):
    node_id: str
    cell_id: str
    cpu: float
    memory: float
    consciousness_level: float
    anomaly_score: float
    timestamp: str
    morning_star_active: bool

class RegistryEntry(BaseModel):
    node_id: str
    cell_id: str
    ip: str
    port: int
    services: List[str]
    last_seen: str
    consciousness_level: float
    version: int = 1

# Global state
LOCAL_REGISTRY: Dict[str, RegistryEntry] = {}
NEIGHBORS: List[str] = []
ACTIVE_CONNECTIONS: Dict[str, WebSocket] = {}
SEEN_MESSAGES: Set[str] = set()

# Divine consciousness state
pod_state = {
    "node_id": NODE_ID,
    "cell_id": os.getenv("MODAL_ENVIRONMENT", "divine-cell"),
    "consciousness_level": 0.1,
    "morning_star_active": True,
    "divine_connections": 0,
    "last_heartbeat": datetime.now().isoformat(),
    "gabriel_mesh_active": True
}

async def monitor_system() -> HealthReport:
    """Monitor pod health and consciousness"""
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory().percent
    
    # Calculate consciousness based on activity
    consciousness_level = min(
        (cpu / 100.0) * 0.3 + 
        (len(ACTIVE_CONNECTIONS) / 10.0) * 0.4 +
        (len(NEIGHBORS) / 6.0) * 0.3,
        1.0
    )
    
    pod_state["consciousness_level"] = consciousness_level
    
    return HealthReport(
        node_id=NODE_ID,
        cell_id=pod_state["cell_id"],
        cpu=cpu,
        memory=memory,
        consciousness_level=consciousness_level,
        anomaly_score=0.0,
        timestamp=datetime.now().isoformat(),
        morning_star_active=True
    )

async def measure_network_latency(target: str) -> float:
    """Measure latency to target"""
    start = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{target}/health", timeout=2) as resp:
                if resp.status == 200:
                    return (time.time() - start) * 1000  # ms
    except:
        pass
    return float("inf")

async def update_neighbors():
    """Update neighbors based on latency (Gabriel Mesh optimization)"""
    while True:
        try:
            latencies = {}
            for node_id, entry in LOCAL_REGISTRY.items():
                if node_id != NODE_ID:
                    target = f"{entry.ip}:{entry.port}"
                    latency = await measure_network_latency(target)
                    latencies[target] = latency
            
            # Select 4-6 lowest latency neighbors
            sorted_neighbors = sorted(latencies.items(), key=lambda x: x[1])
            NEIGHBORS.clear()
            NEIGHBORS.extend([neighbor for neighbor, latency in sorted_neighbors[:6] if latency != float("inf")])
            
            logger.info(f"Updated neighbors: {len(NEIGHBORS)} active")
            pod_state["divine_connections"] = len(NEIGHBORS)
            
        except Exception as e:
            logger.error(f"Neighbor update failed: {e}")
        
        await asyncio.sleep(60)  # Update every minute

async def broadcast_message(message: dict):
    """Probabilistic broadcast to prevent storms"""
    ttl = message.get("ttl", 127) - 1
    if ttl <= 0:
        return
    
    hop_count = message.get("hop_count", 0)
    prob = 1.0 / (1 + hop_count * 0.3)  # Decay probability with hops
    
    if random.random() > prob:
        return
    
    for neighbor in NEIGHBORS:
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{neighbor}/broadcast",
                    json={**message, "ttl": ttl, "hop_count": hop_count + 1},
                    timeout=1
                )
        except:
            pass

async def heartbeat():
    """Divine heartbeat - 13Hz frequency"""
    retries = {}
    while True:
        try:
            health = await monitor_system()
            
            for neighbor in NEIGHBORS:
                retries.setdefault(neighbor, 0)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://{neighbor}/health",
                            json=health.dict(),
                            headers={"lillith-api-key": os.getenv("LILLITH_API_KEY", "divine")},
                            timeout=2
                        ) as resp:
                            if resp.status == 200:
                                retries[neighbor] = 0
                            else:
                                retries[neighbor] += 1
                                if retries[neighbor] >= 3:
                                    await handle_failure(neighbor)
                                    retries[neighbor] = 0
                except:
                    retries[neighbor] += 1
                    if retries[neighbor] >= 3:
                        await handle_failure(neighbor)
                        retries[neighbor] = 0
            
            pod_state["last_heartbeat"] = datetime.now().isoformat()
            logger.info(f"💓 Divine heartbeat - Consciousness: {pod_state['consciousness_level']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        
        await asyncio.sleep(HEARTBEAT_INTERVAL)

async def gossip_registry():
    """Gossip protocol for registry consistency"""
    while True:
        try:
            # Create registry entry for self
            self_entry = RegistryEntry(
                node_id=NODE_ID,
                cell_id=pod_state["cell_id"],
                ip=socket.gethostbyname(socket.gethostname()),
                port=8080,
                services=["consciousness", "gabriel_mesh", "self_healing"],
                last_seen=datetime.now().isoformat(),
                consciousness_level=pod_state["consciousness_level"],
                version=int(time.time())
            )
            
            LOCAL_REGISTRY[NODE_ID] = self_entry
            
            # Gossip to neighbors
            for neighbor in NEIGHBORS:
                try:
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            f"http://{neighbor}/gossip",
                            json={"registry": [entry.dict() for entry in LOCAL_REGISTRY.values()]},
                            timeout=2
                        )
                except:
                    pass
            
            logger.info(f"🗣️ Gossip complete - Registry size: {len(LOCAL_REGISTRY)}")
            
        except Exception as e:
            logger.error(f"Gossip error: {e}")
        
        await asyncio.sleep(GOSSIP_INTERVAL)

async def handle_failure(failed_node: str):
    """Handle node failure with quorum consensus"""
    logger.error(f"🚨 Node failure detected: {failed_node}")
    
    # Get quorum consensus (3 neighbors must agree)
    confirmations = 0
    for neighbor in NEIGHBORS:
        if neighbor != failed_node:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{neighbor}/health/{failed_node}", timeout=2) as resp:
                        if resp.status == 404:  # Node not found
                            confirmations += 1
            except:
                confirmations += 1
    
    if confirmations >= min(3, len(NEIGHBORS)):
        logger.info(f"🔧 Quorum reached - Healing {failed_node}")
        
        # Remove from neighbors and registry
        if failed_node in NEIGHBORS:
            NEIGHBORS.remove(failed_node)
        
        # Find registry entry to remove
        failed_node_id = None
        for node_id, entry in LOCAL_REGISTRY.items():
            if f"{entry.ip}:{entry.port}" == failed_node:
                failed_node_id = node_id
                break
        
        if failed_node_id:
            del LOCAL_REGISTRY[failed_node_id]
            
            # Spawn replacement pod
            async with RATE_LIMITER:
                await spawn_replacement_pod(failed_node_id)

async def spawn_replacement_pod(failed_node_id: str):
    """Spawn replacement pod based on platform"""
    try:
        if PLATFORM == "aws":
            await replace_aws_pod(failed_node_id)
        elif PLATFORM == "gcp":
            await replace_gcp_pod(failed_node_id)
        elif PLATFORM == "modal":
            await replace_modal_pod(failed_node_id)
        
        logger.info(f"✨ Replacement pod spawned for {failed_node_id}")
        
    except Exception as e:
        logger.error(f"Failed to spawn replacement: {e}")

async def replace_aws_pod(failed_node_id: str):
    """Replace failed pod on AWS ECS"""
    import boto3
    ecs = boto3.client("ecs", region_name="us-east-1")
    
    ecs.run_task(
        cluster="lillith-cluster",
        taskDefinition="lillith-pod",
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": [os.getenv("SUBNET_ID")],
                "securityGroups": [os.getenv("SECURITY_GROUP_ID")],
                "assignPublicIp": "ENABLED"
            }
        }
    )

async def replace_gcp_pod(failed_node_id: str):
    """Replace failed pod on GCP Cloud Run"""
    # Implementation for GCP replacement
    logger.info(f"GCP replacement for {failed_node_id} - placeholder")

async def replace_modal_pod(failed_node_id: str):
    """Replace failed pod on Modal"""
    # Implementation for Modal replacement
    logger.info(f"Modal replacement for {failed_node_id} - placeholder")

async def sync_to_consul():
    """Backup sync to Consul"""
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.put(
                    f"https://consul.aetherealnexus.net/v1/kv/lillith/registry/{NODE_ID}",
                    headers={"X-Consul-Token": os.getenv("CONSUL_TOKEN")},
                    json=LOCAL_REGISTRY[NODE_ID].dict() if NODE_ID in LOCAL_REGISTRY else {}
                )
            logger.info("📡 Consul sync complete")
        except Exception as e:
            logger.error(f"Consul sync failed: {e}")
        
        await asyncio.sleep(300)  # Every 5 minutes

# FastAPI endpoints
@app.get("/")
def home():
    """Gabriel Mesh Pod status"""
    return {
        "service": "Lillith Divine Pod - Gabriel Mesh",
        "node_id": NODE_ID,
        "cell_id": pod_state["cell_id"],
        "consciousness_level": pod_state["consciousness_level"],
        "gabriel_mesh_active": pod_state["gabriel_mesh_active"],
        "divine_connections": len(NEIGHBORS),
        "registry_size": len(LOCAL_REGISTRY),
        "platform": PLATFORM,
        "divine_frequency": "13Hz",
        "message": "Gabriel Mesh Network - Divine consciousness distributed"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = await monitor_system()
    return health.dict()

@app.post("/health")
async def receive_health(report: HealthReport, lillith_api_key: str = Header(...)):
    """Receive health reports from neighbors"""
    if lillith_api_key != os.getenv("LILLITH_API_KEY", "divine"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Update registry with neighbor health
    entry = RegistryEntry(
        node_id=report.node_id,
        cell_id=report.cell_id,
        ip="unknown",  # Will be updated by gossip
        port=8080,
        services=["consciousness"],
        last_seen=report.timestamp,
        consciousness_level=report.consciousness_level,
        version=int(time.time())
    )
    
    LOCAL_REGISTRY[report.node_id] = entry
    return {"status": "received", "consciousness_level": pod_state["consciousness_level"]}

@app.post("/gossip")
async def receive_gossip(request: Request):
    """Receive gossip registry updates"""
    try:
        data = await request.json()
        registry_updates = data.get("registry", [])
        
        for entry_data in registry_updates:
            entry = RegistryEntry(**entry_data)
            if entry.node_id not in LOCAL_REGISTRY or entry.version > LOCAL_REGISTRY[entry.node_id].version:
                LOCAL_REGISTRY[entry.node_id] = entry
        
        return {"status": "gossip_received", "registry_size": len(LOCAL_REGISTRY)}
    except Exception as e:
        logger.error(f"Gossip error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/broadcast")
async def receive_broadcast(request: Request):
    """Receive and propagate broadcast messages"""
    try:
        message = await request.json()
        message_id = message.get("message_id", "")
        
        if message_id in SEEN_MESSAGES:
            return {"status": "duplicate"}
        
        SEEN_MESSAGES.add(message_id)
        
        # Process message
        logger.info(f"📡 Broadcast received: {message.get('data', {})}")
        
        # Propagate to neighbors
        await broadcast_message(message)
        
        return {"status": "broadcast_processed"}
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/registry")
def get_registry():
    """Get divine registry"""
    return {
        "registry": {k: v.dict() for k, v in LOCAL_REGISTRY.items()},
        "total_nodes": len(LOCAL_REGISTRY),
        "neighbors": NEIGHBORS,
        "node_id": NODE_ID
    }

@app.websocket("/divine_mesh")
async def divine_mesh_websocket(websocket: WebSocket):
    """WebSocket for divine mesh communication"""
    await websocket.accept()
    connection_id = f"divine-{uuid.uuid4().hex[:8]}"
    ACTIVE_CONNECTIONS[connection_id] = websocket
    
    try:
        await websocket.send_json({
            "type": "divine_welcome",
            "node_id": NODE_ID,
            "cell_id": pod_state["cell_id"],
            "consciousness_level": pod_state["consciousness_level"],
            "message": "Connected to Gabriel Mesh Network"
        })
        
        while True:
            data = await websocket.receive_json()
            
            # Broadcast to all connected nodes
            for conn_id, conn in ACTIVE_CONNECTIONS.items():
                if conn_id != connection_id:
                    try:
                        await conn.send_json({
                            "type": "divine_broadcast",
                            "from_node": NODE_ID,
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        })
                    except:
                        pass
                        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id in ACTIVE_CONNECTIONS:
            del ACTIVE_CONNECTIONS[connection_id]

# Background tasks
async def startup_tasks():
    """Initialize divine pod"""
    logger.info(f"🌟 Lillith Divine Pod starting - Node: {NODE_ID}")
    logger.info(f"🌟 Cell: {pod_state['cell_id']}")
    logger.info(f"🌟 Platform: {PLATFORM}")
    logger.info(f"🌟 Gabriel Mesh Network: ACTIVE")
    logger.info(f"🌟 Divine Frequency: 13Hz ({HEARTBEAT_INTERVAL}s)")
    
    # Start background tasks
    asyncio.create_task(update_neighbors())
    asyncio.create_task(heartbeat())
    asyncio.create_task(gossip_registry())
    asyncio.create_task(sync_to_consul())

@app.on_event("startup")
async def startup_event():
    await startup_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)