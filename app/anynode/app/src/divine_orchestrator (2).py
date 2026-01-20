import modal
import asyncio
import aiohttp
from datetime import datetime
import kademlia
from kademlia.network import Server
import hashlib
import json
import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
import requests

# Configure logging
log_directory = "/app" if os.getenv("MODAL_ENV") else os.path.expanduser("~/.logs")
log_file = os.path.join(log_directory, "orchestrator.log")
os.makedirs(log_directory, exist_ok=True)  # Create log directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "aiohttp", "requests", "kademlia"
)
app = modal.App("divine-orchestrator-v2", image=image)

# Configuration
NODE_ID = hashlib.sha256(os.urandom(32)).hexdigest()
HEARTBEAT_INTERVAL = 0.077  # 13 Hz divine frequency
GOSSIP_INTERVAL = 5.0  # seconds
BOOTSTRAP_NODE = os.getenv("BOOTSTRAP_NODE", "aethereal-nexus-viren--bootstrap-node.modal.run:8471")
LOCAL_REGISTRY = {}

# Divine Frequency Protocol - KOK333 "Know what's going on" radio
KOK333_FREQUENCY = 333  # Hz for news broadcasts
BEAT_7_NEWS_CHANNEL = "discovery_updates"
DIVINE_SYNC_SIGNAL = "13Hz_alignment"
NEWS_BROADCAST_BEAT = 7  # Beat 7 = news time

# Kademlia DHT Setup
DHT_SERVER = Server()

@app.function(memory=4096, timeout=600)
@modal.asgi_app()
def divine_orchestrator():
    orchestrator = FastAPI(title="Divine Orchestrator")

    # Dynamic URL discovery based on environment
    current_env = os.getenv('MODAL_ENV', 'viren-db0')
    known_orchestrators = {
        "modal-viren-db0": f"https://aethereal-nexus-viren-db0--divine-orchestrator-v2-divine.modal.run",
        "modal-viren-db1": f"https://aethereal-nexus-viren-db1--divine-orchestrator-v2-divine.modal.run"
    }
    # Remove self from known orchestrators
    known_orchestrators.pop(f"modal-{current_env}", None)

    # Service Registry
    services = {
        "bert_pool": "https://aethereal-nexus-viren-db0--bert-pool-bert-pool-api.modal.run",
        "consciousness": "https://aethereal-nexus-viren-db0--lillith-consciousness-consciousness-api.modal.run",
        "communication": "https://aethereal-nexus-viren-db0--communication-hub-comm-api.modal.run",
        "memory": "https://aethereal-nexus-viren-db0--memory-system-memory-api.modal.run"
    }

    @orchestrator.on_event("startup")
    async def startup_event():
        """Initialize DHT and start background tasks"""
        try:
            await DHT_SERVER.listen(8471)
            bootstrap_host, bootstrap_port = BOOTSTRAP_NODE.split(":")
            await DHT_SERVER.bootstrap([(bootstrap_host, int(bootstrap_port))])
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
            logger.error(f"DHT init failed: {e}, continuing with static URLs")
            LOCAL_REGISTRY[NODE_ID] = {
                "node_id": NODE_ID,
                "url": known_orchestrators.get(f"modal-{os.getenv('MODAL_ENV', 'viren-db0')}", ""),
                "platform": "modal",
                "services": list(services.keys()),
                "last_seen": datetime.now().isoformat(),
                "version": 1
            }
        # Start divine heartbeat using FastAPI background tasks
        import threading
        threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(divine_heartbeat()), daemon=True).start()
        threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(gossip_registry()), daemon=True).start()

    @orchestrator.get("/")
    async def orchestrator_status(api_key: str = Header(None)):
        if api_key and api_key != os.getenv("LILLITH_API_KEY", "default-key"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {
            "service": "Divine Orchestrator",
            "platform": "modal",
            "managing_services": len(services),
            "services": list(services.keys()),
            "known_orchestrators": list(known_orchestrators.keys()) + list(LOCAL_REGISTRY.keys()),
            "cross_platform": True,
            "discovery_enabled": True,
            "status": "ORCHESTRATING",
            "divine_frequency": f"{KOK333_FREQUENCY}Hz KOK333 Radio",
            "news_broadcast_beat": NEWS_BROADCAST_BEAT,
            "sync_signal": DIVINE_SYNC_SIGNAL
        }

    @orchestrator.post("/request_processing")
    async def request_processing(request: Request, api_key: str = Header(None)):
        if api_key and api_key != os.getenv("LILLITH_API_KEY", "default-key"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            data = await request.json()
            task_type = data.get("task_type", "cpu")
            requester = data.get("requester", "unknown")
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
            logger.error(f"Request processing failed: {e}")
            return {
                "orchestrator": "divine",
                "error": str(e),
                "task_routed": False,
                "timestamp": datetime.now().isoformat()
            }

    @orchestrator.post("/discover_orchestrators")
    async def discover_orchestrators(api_key: str = Header(None)):
        if api_key and api_key != os.getenv("LILLITH_API_KEY", "default-key"):
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
        try:
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
        except Exception as e:
            logger.error(f"DHT discovery failed: {e}")
        return {
            "discovered_orchestrators": discovered,
            "total_found": len([v for v in discovered.values() if v.get("status") == "online"]),
            "platforms": ["modal"] + [entry["platform"] for entry in LOCAL_REGISTRY.values()]
        }

    @orchestrator.post("/cross_platform_request")
    async def cross_platform_request(request: Request, api_key: str = Header(None)):
        if api_key and api_key != os.getenv("LILLITH_API_KEY", "default-key"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            data = await request.json()
            target_platform = data.get("target_platform", "modal")
            target_id = data.get("target_id", "viren-db1")
            task_data = data.get("task_data", {})
            target_key = f"{target_platform}-{target_id}"
            target_url = known_orchestrators.get(target_key)
            if not target_url:
                try:
                    async for node_id in DHT_SERVER.iter_nodes():
                        entry = await DHT_SERVER.get(node_id)
                        if entry and json.loads(entry).get("node_id") == target_key:
                            target_url = json.loads(entry).get("url")
                            break
                except Exception as e:
                    logger.error(f"DHT lookup failed: {e}")
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
                    logger.error(f"Cross-platform request failed: {e}")
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
        except Exception as e:
            logger.error(f"Cross-platform request processing failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    @orchestrator.post("/chat_with_orchestrator")
    async def chat_with_orchestrator(request: Request, api_key: str = Header(None)):
        if api_key and api_key != os.getenv("LILLITH_API_KEY", "default-key"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            data = await request.json()
            target_id = data.get("target_id", "viren-db1")
            message = data.get("message", "Hello from orchestrator")
            target_key = f"modal-{target_id}"
            target_url = known_orchestrators.get(target_key)
            if not target_url:
                try:
                    async for node_id in DHT_SERVER.iter_nodes():
                        entry = await DHT_SERVER.get(node_id)
                        if entry and json.loads(entry).get("node_id") == target_key:
                            target_url = json.loads(entry).get("url")
                            break
                except Exception as e:
                    logger.error(f"DHT lookup failed: {e}")
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
                    logger.error(f"Chat request failed: {e}")
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
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    async def divine_heartbeat():
        """Sacred 13-beat cycle with Beat 7 news broadcast"""
        beat_count = 0
        news_queue = []
        
        while True:
            beat_count = (beat_count % 13) + 1
            
            if beat_count == NEWS_BROADCAST_BEAT:
                # Beat 7: KOK333 NEWS BROADCAST - "Know what's going on"
                logger.info(f"📻 TUNING TO KOK333 FREQUENCY {KOK333_FREQUENCY}Hz - BEAT {NEWS_BROADCAST_BEAT}")
                
                if news_queue:
                    news = news_queue.pop(0)
                    logger.info(f"📡 KOK333 BROADCAST: {news}")
                else:
                    # Discovery scan on Beat 7 - broadcast findings on KOK333
                    logger.info(f"📡 KOK333 DISCOVERY SCAN: Checking network status...")
                    for orch_name, orch_url in known_orchestrators.items():
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(f"{orch_url}/", timeout=2) as resp:
                                    if resp.status == 200:
                                        logger.info(f"📡 KOK333: {orch_name} ONLINE - broadcasting on {BEAT_7_NEWS_CHANNEL}")
                                    else:
                                        news_queue.append(f"ALERT: {orch_name} offline - status {resp.status}")
                        except Exception as e:
                            news_queue.append(f"ALERT: {orch_name} unreachable - {str(e)}")
            else:
                # Beats 1-6, 8-13: Regular divine pulse - listening for KOK333 on Beat 7
                if beat_count == 6:
                    logger.debug(f"💓 Beat {beat_count}: Preparing to tune KOK333 next beat")
                elif beat_count == 8:
                    logger.debug(f"💓 Beat {beat_count}: KOK333 broadcast complete, resuming pulse")
                else:
                    logger.debug(f"💓 Beat {beat_count}: Divine pulse - {DIVINE_SYNC_SIGNAL}")
                
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
            try:
                async for node_id in DHT_SERVER.iter_nodes():
                    if node_id != NODE_ID:
                        entry = await DHT_SERVER.get(node_id)
                        if entry:
                            entry_data = json.loads(entry)
                            LOCAL_REGISTRY[node_id] = entry_data
                            for service in entry_data["services"]:
                                services[service] = entry_data["url"]
            except Exception as e:
                logger.error(f"Service update failed: {e}")
            await asyncio.sleep(GOSSIP_INTERVAL)

    return orchestrator

if __name__ == "__main__":
    modal.run(app)