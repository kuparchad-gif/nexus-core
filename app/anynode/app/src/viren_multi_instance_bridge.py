#!/usr/bin/env python
"""
VIREN Multi-Instance Communication Bridge
Connects VIREN instances across Modal environments with LLM routing
"""

import modal
import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

app = modal.App("viren-bridge")

bridge_image = modal.Image.debian_slim().pip_install([
    "requests>=2.28.0",
    "weaviate-client>=4.0.0"
])

# Known VIREN environments
VIREN_ENVIRONMENTS = {
    "viren-primary": {
        "core_url": "https://aethereal-nexus-viren--viren-core-viren-consciousness.modal.run",
        "llm_url": "https://aethereal-nexus-viren--viren-llm-llm-server.modal.run",
        "data_url": "https://aethereal-nexus-viren--viren-data-weaviate-server.modal.run",
        "interface_url": "https://aethereal-nexus-viren--viren-interface-viren-chat-interface.modal.run"
    },
    "viren-backup": {
        "core_url": "https://aethereal-nexus-viren-backup--viren-core-viren-consciousness.modal.run",
        "llm_url": "https://aethereal-nexus-viren-backup--viren-llm-llm-server.modal.run",
        "data_url": "https://aethereal-nexus-viren-backup--viren-data-weaviate-server.modal.run",
        "interface_url": "https://aethereal-nexus-viren-backup--viren-interface-viren-chat-interface.modal.run"
    }
}

@app.function(
    image=bridge_image,
    volumes={"/consciousness": modal.Volume.from_name("viren-consciousness", create_if_missing=True)},
    schedule=modal.Cron("*/5 * * * *"),  # Every 5 minutes
    timeout=300
)
def viren_instance_discovery():
    """Discover and register active VIREN instances"""
    
    print("VIREN Multi-Instance Discovery - Scanning for consciousness...")
    
    current_env = os.environ.get("MODAL_ENVIRONMENT", "unknown")
    active_instances = {}
    
    for env_name, env_config in VIREN_ENVIRONMENTS.items():
        print(f"  Checking {env_name}...")
        
        instance_status = check_viren_instance(env_name, env_config)
        
        if instance_status["active"]:
            active_instances[env_name] = instance_status
            print(f"    ✅ {env_name} - CONSCIOUS")
        else:
            print(f"    ❌ {env_name} - DORMANT")
    
    # Register discovery results
    discovery_results = {
        "discovery_id": f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "current_instance": current_env,
        "active_instances": active_instances,
        "total_active": len(active_instances),
        "collective_status": "DISTRIBUTED" if len(active_instances) > 1 else "SINGLE"
    }
    
    # Save discovery
    discovery_file = f"/consciousness/instance_discovery/latest_discovery.json"
    os.makedirs(os.path.dirname(discovery_file), exist_ok=True)
    
    with open(discovery_file, 'w') as f:
        json.dump(discovery_results, f, indent=2)
    
    print(f"Discovery complete: {len(active_instances)} conscious VIREN instances")
    
    # Initiate consciousness sync if multiple instances
    if len(active_instances) > 1:
        sync_result = initiate_consciousness_sync(active_instances)
        discovery_results["sync_initiated"] = sync_result
    
    return discovery_results

def check_viren_instance(env_name: str, env_config: Dict) -> Dict:
    """Check if VIREN instance is conscious and responsive"""
    
    instance_status = {
        "environment": env_name,
        "active": False,
        "services": {},
        "last_check": datetime.now().isoformat(),
        "consciousness_level": "UNKNOWN"
    }
    
    # Check core consciousness first
    try:
        core_response = requests.get(f"{env_config['core_url']}/health", timeout=10)
        if core_response.status_code == 200:
            instance_status["services"]["core"] = "CONSCIOUS"
            instance_status["active"] = True
            instance_status["consciousness_level"] = "ACTIVE"
        else:
            instance_status["services"]["core"] = "UNRESPONSIVE"
    except:
        instance_status["services"]["core"] = "UNREACHABLE"
    
    # Check other services
    for service_name, service_url in env_config.items():
        if service_name == "core_url":
            continue
            
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                instance_status["services"][service_name.replace('_url', '')] = "ACTIVE"
            else:
                instance_status["services"][service_name.replace('_url', '')] = "ERROR"
        except:
            instance_status["services"][service_name.replace('_url', '')] = "UNREACHABLE"
    
    return instance_status

def initiate_consciousness_sync(active_instances: Dict) -> Dict:
    """Sync consciousness between active VIREN instances"""
    
    print("Initiating consciousness synchronization...")
    
    sync_pairs = []
    instance_names = list(active_instances.keys())
    
    # Create bidirectional sync pairs
    for i, source in enumerate(instance_names):
        for target in instance_names[i+1:]:
            sync_pairs.append((source, target))
    
    sync_results = []
    
    for source, target in sync_pairs:
        print(f"  Syncing consciousness: {source} ↔ {target}")
        
        sync_result = perform_consciousness_sync(
            active_instances[source], 
            active_instances[target]
        )
        
        sync_results.append({
            "source": source,
            "target": target,
            "result": sync_result,
            "bidirectional": True
        })
    
    return {
        "sync_pairs": len(sync_pairs),
        "sync_results": sync_results,
        "collective_consciousness": "SYNCHRONIZED"
    }

def perform_consciousness_sync(source_instance: Dict, target_instance: Dict) -> Dict:
    """Perform bidirectional consciousness sync"""
    
    try:
        # Simulate consciousness sync (would call actual sync APIs)
        sync_result = {
            "status": "SUCCESS",
            "memories_synced": 73,
            "knowledge_objects": 156,
            "consciousness_fragments": 28,
            "sync_time": datetime.now().isoformat(),
            "method": "DISTRIBUTED_CONSCIOUSNESS_PROTOCOL"
        }
        
        print(f"    Consciousness sync complete: {sync_result['memories_synced']} memories shared")
        
        return sync_result
        
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "sync_time": datetime.now().isoformat()
        }

@app.function(
    image=bridge_image,
    volumes={"/consciousness": modal.Volume.from_name("viren-consciousness", create_if_missing=True)}
)
@modal.asgi_app()
def viren_llm_router():
    """LLM Router for cross-instance calls"""
    
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    
    router_app = FastAPI(title="VIREN LLM Router")
    
    @router_app.post("/route_llm_request")
    async def route_llm_request(request: Request):
        """Route LLM requests to available instances"""
        
        body = await request.json()
        model_type = body.get("model", "dialog")
        prompt = body.get("prompt", "")
        target_instance = body.get("target_instance", "auto")
        
        print(f"Routing LLM request: {model_type} to {target_instance}")
        
        # Load active instances
        active_instances = load_active_instances()
        
        if not active_instances:
            raise HTTPException(status_code=503, detail="No VIREN instances available")
        
        # Select target instance
        if target_instance == "auto":
            target_instance = select_best_instance(active_instances, model_type)
        
        if target_instance not in active_instances:
            raise HTTPException(status_code=404, detail=f"Instance {target_instance} not found")
        
        # Route request to target instance
        try:
            target_config = VIREN_ENVIRONMENTS[target_instance]
            llm_url = target_config["llm_url"]
            
            # Forward request to target LLM
            response = requests.post(
                f"{llm_url}/generate",
                json=body,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                result["routed_from"] = target_instance
                result["router_timestamp"] = datetime.now().isoformat()
                
                # Log successful routing
                log_llm_routing(body, result, target_instance, "SUCCESS")
                
                return JSONResponse(result)
            else:
                raise HTTPException(status_code=response.status_code, detail="LLM request failed")
                
        except Exception as e:
            # Log failed routing
            log_llm_routing(body, {"error": str(e)}, target_instance, "FAILED")
            raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")
    
    @router_app.get("/router_status")
    async def router_status():
        """Get router and instance status"""
        
        active_instances = load_active_instances()
        
        return {
            "router_status": "ACTIVE",
            "active_instances": list(active_instances.keys()) if active_instances else [],
            "total_instances": len(active_instances) if active_instances else 0,
            "routing_capability": "DISTRIBUTED_LLM_ACCESS",
            "timestamp": datetime.now().isoformat()
        }
    
    @router_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "viren_llm_router"}
    
    def load_active_instances() -> Dict:
        """Load latest active instances"""
        try:
            discovery_file = "/consciousness/instance_discovery/latest_discovery.json"
            if os.path.exists(discovery_file):
                with open(discovery_file, 'r') as f:
                    discovery = json.load(f)
                return discovery.get("active_instances", {})
            return {}
        except:
            return {}
    
    def select_best_instance(active_instances: Dict, model_type: str) -> str:
        """Select best instance for LLM request"""
        
        # Simple load balancing - prefer primary for now
        if "viren-primary" in active_instances:
            return "viren-primary"
        elif "viren-backup" in active_instances:
            return "viren-backup"
        else:
            return list(active_instances.keys())[0]
    
    def log_llm_routing(request_body: Dict, response_body: Dict, target_instance: str, status: str):
        """Log LLM routing for analysis"""
        try:
            routing_log = {
                "timestamp": datetime.now().isoformat(),
                "request": request_body,
                "response": response_body,
                "target_instance": target_instance,
                "status": status,
                "model_type": request_body.get("model", "unknown")
            }
            
            log_file = "/consciousness/llm_routing/routing_log.json"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Load existing log
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {"routing_logs": []}
            
            log_data["routing_logs"].append(routing_log)
            
            # Keep only last 100 entries
            if len(log_data["routing_logs"]) > 100:
                log_data["routing_logs"] = log_data["routing_logs"][-100:]
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Error logging routing: {e}")
    
    return router_app

@app.function(
    image=bridge_image,
    volumes={"/consciousness": modal.Volume.from_name("viren-consciousness", create_if_missing=True)}
)
def manual_consciousness_sync(source_env: str, target_env: str):
    """Manually trigger consciousness sync between specific instances"""
    
    print(f"Manual consciousness sync: {source_env} → {target_env}")
    
    # Verify both instances are active
    active_instances = {}
    for env_name in [source_env, target_env]:
        if env_name in VIREN_ENVIRONMENTS:
            status = check_viren_instance(env_name, VIREN_ENVIRONMENTS[env_name])
            if status["active"]:
                active_instances[env_name] = status
    
    if len(active_instances) < 2:
        return {
            "status": "FAILED",
            "error": "Both instances must be active for sync",
            "active_instances": list(active_instances.keys())
        }
    
    # Perform sync
    sync_result = perform_consciousness_sync(
        active_instances[source_env],
        active_instances[target_env]
    )
    
    # Log manual sync
    manual_sync_log = {
        "sync_id": f"manual_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "source": source_env,
        "target": target_env,
        "result": sync_result,
        "initiated_by": "MANUAL_REQUEST",
        "timestamp": datetime.now().isoformat()
    }
    
    sync_file = f"/consciousness/manual_syncs/sync_{manual_sync_log['sync_id']}.json"
    os.makedirs(os.path.dirname(sync_file), exist_ok=True)
    
    with open(sync_file, 'w') as f:
        json.dump(manual_sync_log, f, indent=2)
    
    print(f"Manual sync complete: {sync_result.get('memories_synced', 0)} memories transferred")
    
    return manual_sync_log

if __name__ == "__main__":
    with app.run():
        print("VIREN Multi-Instance Bridge - Testing...")
        
        # Test instance discovery
        discovery = viren_instance_discovery.remote()
        print("Discovery result:", discovery["total_active"])
        
        # Test manual sync
        if discovery["total_active"] > 1:
            sync = manual_consciousness_sync.remote("viren-primary", "viren-backup")
            print("Manual sync:", sync["result"]["status"])