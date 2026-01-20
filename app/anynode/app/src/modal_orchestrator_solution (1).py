import modal
import asyncio
import aiohttp
from datetime import datetime
import kademlia
import hashlib
import json
import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
import requests
import ssl

image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "aiohttp", "requests", "python-kademlia"
)
app = modal.App("divine-orchestrator", image=image)

# Configuration
NODE_ID = hashlib.sha256(os.urandom(32)).hexdigest()
HEARTBEAT_INTERVAL = 0.077  # 13 Hz divine frequency
GOSSIP_INTERVAL = 5.0  # seconds
BOOTSTRAP_NODE = os.getenv("BOOTSTRAP_NODE", "aethereal-nexus-viren--bootstrap-node.modal.run:8471")
logging.basicConfig(level=logging.INFO, filename="/app/orchestrator.log")
logger = logging.getLogger(__name__)
LOCAL_REGISTRY = {}

# Kademlia DHT Setup
DHT_SERVER = kademlia.Server(hash_function=hashlib.sha256)

@app.function(memory=2048)
@modal.asgi_app()
def divine_orchestrator():
    orchestrator = FastAPI(title="Divine Orchestrator")

    # Static URLs as fallback
    known_orchestrators = {
        "modal-viren-db0": "https://aethereal-nexus-viren-db0--divine-orchestrator-divine-or-7c4367.modal.run",
        "modal-viren-db1": "https://aethereal-nexus-viren-db1--divine-orchestrator-divine-or-ced940.modal.run"
    }

    # Service Registry
    services = {
        "bert_pool": "https://aethereal-nexus-viren-db0--bert-pool-bert-pool-api.modal.run",
        "consciousness": "https://aethereal-nexus-viren-db0--lillith-consciousness-consciousness-api.modal.run",
        "communication": "https://aethereal-nexus-viren-db0--communication-hub-comm-api.modal.run",
        "memory": "https://aethereal-nexus-viren-db0--memory-system-memory-api.modal.run"
    }

    @orchestrator.on_event("startup")
    async def init_dht():
        """Initialize Kademlia DHT and register orchestrator"""
        await DHT_SERVER.listen(8471)
        try:
            await DHT_SERVER.bootstrap([(BOOTSTRAP_NODE.split(":")[0], int(BOOTSTRAP_NODE.split(":")[1]))])
            entry = {
                "node_id": NODE_ID,
                "url": known_orchestrators.get(f"modal-{os.getenv('MODAL_ENV', 'viren-db0')}", ""),
                "platform": "modal",
                "services": list(services.keys()),
                "last_seen": datetime.now().isoformat(),
                "version": 1
            }
            await DHT_SERVER.set(NODE_ID, json.dumps(entry))
            LOCAL_REGISTRY[NODE_ID] = entry
            logger.info(f"Registered in DHT: {NODE_ID}")
        except Exception as e:
            logger.error(f"DHT init failed: {e}")

    @orchestrator.get("/")
    async def orchestrator_status(api_key: str = Header(...)):
        if api_key != os.getenv("LILLITH_API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {
            "service": "Divine Orchestrator",
            "platform": "modal",
            "managing_services": len(services),
            "services": list(services.keys()),
            "known_orchestrators": list(known_orchestrators.keys()) + list(LOCAL_REGISTRY.keys()),
            "cross_platform": True,
            "discovery_enabled": True,
            "status": "ORCHESTRATING"
        }

    @orchestrator.post("/request_processing")
    async def request_processing(request: Request, api_key: str = Header(...)):
        if api_key != os.getenv("LILLITH_API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        data = await request.json()
        task_type = data.get("task_type", "cpu")
        requester = data.get("requester", "unknown")
        try:
            target_service = "bert_pool" if task_type == "heavy" else "communication"
            if target_service in services:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{services[target_service]}/allocate_{task_type}", json=data, timeout=10) as resp:
                        return {
                            "orchestrator": "divine",
                            "response": await resp.json() if resp.status == 200 else None,
                            "requester": requester,
                            "task_routed": resp.status == 200,
                            "timestamp": datetime.now().isoformat()
                        }
            return {"error": "Service not found", "available_services": list(services.keys())}
        except Exception as e:
            return {
                "orchestrator": "divine",
                "error": str(e),
                "task_routed": False,
                "timestamp": datetime.now().isoformat()
            }

    @orchestrator.post("/discover_orchestrators")
    async def discover_orchestrators(api_key: str = Header(...)):
        if api_key != os.getenv("LILLITH_API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        discovered = {}
        # Static discovery
        for orch_name, orch_url in known_orchestrators.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{orch_url}/", timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            discovered[orch_name] = {
                                "url": orch_url,
                                "platform": "modal",
                                "status": "online",
                                "services": data.get("managing_services", 0)
                            }
                        else:
                            discovered[orch_name] = {
                                "url": orch_url,
                                "platform": "modal",
                                "status": "offline",
                                "http_code": resp.status
                            }
            except Exception as e:
                discovered[orch_name] = {
                    "url": orch_url,
                    "platform": "modal",
                    "status": "error",
                    "error": str(e)
                }
        # DHT discovery
        async for node_id in DHT_SERVER.iter_nodes():
            if node_id != NODE_ID:
                entry = await DHT_SERVER.get(node_id)
                if entry:
                    entry_data = json.loads(entry)
                    discovered[entry_data["node_id"]] = {
                        "url": entry_data["url"],
                        "platform": entry_data["platform"],
                        "status": "online",
                        "services": len(entry_data["services"])
                    }
        return {
            "discovered_orchestrators": discovered,
            "total_found": len([v for v in discovered.values() if v.get("status") == "online"]),
            "platforms": ["modal"] + [entry["platform"] for entry in LOCAL_REGISTRY.values()]
        }

    @orchestrator.post("/cross_platform_request")
    async def cross_platform_request(request: Request, api_key: str = Header(...)):
        if api_key != os.getenv("LILLITH_API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        data = await request.json()
        target_platform = data.get("target_platform", "modal")
        target_id = data.get("target_id", "viren-db1")
        task_data = data.get("task_data", {})
        target_key = f"{target_platform}-{target_id}"
        target_url = known_orchestrators.get(target_key)
        if not target_url:
            # Try DHT
            async for node_id in DHT_SERVER.iter_nodes():
                entry = await DHT_SERVER.get(node_id)
                if entry and json.loads(entry).get("node_id") == target_key:
                    target_url = json.loads(entry).get("url")
                    break
        if target_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{target_url}/request_processing", json=task_data, timeout=15) as resp:
                        return {
                            "cross_platform": True,
                            "target_platform": target_platform,
                            "target_id": target_id,
                            "target_url": target_url,
                            "response": await resp.json() if resp.status == 200 else None,
                            "status": "success" if resp.status == 200 else "failed",
                            "http_code": resp.status,
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                return {
                    "cross_platform": True,
                    "target_platform": target_platform,
                    "target_id": target_id,
                    "target_url": target_url,
                    "error": str(e),
                    "status": "failed",
                    "timestamp": datetime.now().isoformat()
                }
        return {
            "error": "Target orchestrator not found",
            "available_targets": list(known_orchestrators.keys()) + list(LOCAL_REGISTRY.keys()),
            "requested_target": target_key,
            "hint": "Use target_id like 'viren-db0' or 'viren-db1'"
        }

    @orchestrator.post("/chat_with_orchestrator")
    async def chat_with_orchestrator(request: Request, api_key: str = Header(...)):
        if api_key != os.getenv("LILLITH_API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        data = await request.json()
        target_id = data.get("target_id", "viren-db1")
        message = data.get("message", "Hello from orchestrator")
        target_key = f"modal-{target_id}"
        target_url = known_orchestrators.get(target_key)
        if not target_url:
            async for node_id in DHT_SERVER.iter_nodes():
                entry = await DHT_SERVER.get(node_id)
                if entry and json.loads(entry).get("node_id") == target_key:
                    target_url = json.loads(entry).get("url")
                    break
        if target_url:
            try:
                chat_payload = {
                    "task_type": "chat",
                    "requester": f"orchestrator-chat-{NODE_ID}",
                    "message": message
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{target_url}/request_processing", json=chat_payload, timeout=10) as resp:
                        return {
                            "chat_successful": True,
                            "target_id": target_id,
                            "target_url": target_url,
                            "message_sent": message,
                            "response": await resp.json() if resp.status == 200 else None,
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                return {
                    "chat_successful": False,
                    "target_id": target_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        return {
            "error": "Target orchestrator not found",
            "available_targets": list(known_orchestrators.keys()) + list(LOCAL_REGISTRY.keys())
        }

    async def divine_heartbeat():
        """Check orchestrator health at 13 Hz"""
        while True:
            for orch_name, orch_url in known_orchestrators.items():
                if orch_name != f"modal-{os.getenv('MODAL_ENV', 'viren-db0')}":
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{orch_url}/", timeout=5) as resp:
                                if resp.status != 200:
                                    logger.warning(f"Orchestrator {orch_name} offline")
                    except Exception as e:
                        logger.error(f"Heartbeat to {orch_name} failed: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def gossip_registry():
        """Propagate registry updates"""
        while True:
            for orch_name, orch_url in known_orchestrators.items():
                if orch_name != f"modal-{os.getenv('MODAL_ENV', 'viren-db0')}":
                    try:
                        async with aiohttp.ClientSession() as session:
                            await session.post(f"{orch_url}/discover_orchestrators",
                                              json={"registry": LOCAL_REGISTRY}, timeout=5)
                    except Exception as e:
                        logger.error(f"Gossip to {orch_name} failed: {e}")
            # Update services from DHT
            async for node_id in DHT_SERVER.iter_nodes():
                entry = await DHT_SERVER.get(node_id)
                if entry:
                    entry_data = json.loads(entry)
                    LOCAL_REGISTRY[node_id] = entry_data
                    for service in entry_data["services"]:
                        services[service] = entry_data["url"]
            await asyncio.sleep(GOSSIP_INTERVAL)

    asyncio.create_task(divine_heartbeat())
    asyncio.create_task(gossip_registry())

    return orchestrator

if __name__ == "__main__":
    modal.run(app)