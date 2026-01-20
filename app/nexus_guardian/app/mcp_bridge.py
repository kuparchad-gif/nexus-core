# mcp_bridge.py
import modal
import asyncio
from typing import Dict, Any
import aiohttp
import json

# Your existing Metatron Router container
METATRON_URL = "metatron-router.modal.run"  # Your running container
COMPACTIFAI_URL = "compactifai-processor.modal.run"  # New container

app = modal.App("mcp-bridge")

class MCPBridge:
    """MCP Bridge between Metatron Router and CompactifAI Processor"""
    
    def __init__(self):
        self.sessions = {}
        
    async def connect_containers(self):
        """Establish MCP connection between the two containers"""
        # Connect to Metatron Router
        self.metatron_session = aiohttp.ClientSession(
            base_url=f"https://{METATRON_URL}",
            headers={"Content-Type": "application/json"}
        )
        
        # Connect to CompactifAI Processor  
        self.compactifai_session = aiohttp.ClientSession(
            base_url=f"https://{COMPACTIFAI_URL}",
            headers={"Content-Type": "application/json"}
        )
        
        print("🔗 MCP Bridge: Containers connected")
        
    async def route_to_compactifai(self, model_data: Dict[str, Any]):
        """Route model processing requests from Metatron to CompactifAI"""
        try:
            async with self.compactifai_session.post("/process", json=model_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ CompactifAI processing started: {result}")
                    return result
                else:
                    print(f"❌ CompactifAI request failed: {resp.status}")
                    return None
                    
        except Exception as e:
            print(f"❌ MCP connection error: {e}")
            return None
            
    async def get_processing_status(self, task_id: str):
        """Check status of processing task"""
        async with self.compactifai_session.get(f"/status/{task_id}") as resp:
            return await resp.json()

@app.function(
    image=modal.Image.debian_slim().pip_install(["aiohttp"]),
    secrets=[modal.Secret.from_name("mcp-bridge-secrets")]
)
async def start_mcp_bridge():
    """Start the MCP bridge between containers"""
    bridge = MCPBridge()
    await bridge.connect_containers()
    
    # Example: Route a model for processing
    model_request = {
        "model_path": "/models/processing/new_model",
        "action": "train_and_compress", 
        "data_path": "/datasets",
        "priority": "high"
    }
    
    result = await bridge.route_to_compactifai(model_request)
    return {"status": "connected", "processing_started": result is not None}