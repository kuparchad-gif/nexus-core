#!/usr/bin/env python3
"""
Integrate Aethereal Nexus with Gabriel's Horn network
"""
import asyncio
from typing import Dict, Any
from gabriels_horn_network import GabrielsHornNetwork

class NexusIntegration:
    def __init__(self):
        self.network = GabrielsHornNetwork()
        asyncio.create_task(self.network.initialize())
    
    async def route_request(self, request: Dict[str, Any], user_id: str = None, ai_name: str = None):
        """Route request through Gabriel's Horn network"""
        if user_id:
            request["user_id"] = user_id
        if ai_name:
            request["ai_name"] = ai_name
        
        return await self.network.route_request(request)
    
    async def register_pod(self, pod_type: str, pod_id: str, config: Dict[str, Any]):
        """Register a pod with the network"""
        await self.network.register_node(pod_id, {
            "type": pod_type,
            **config
        })
        return {"status": "registered", "pod_id": pod_id}

# Usage:
# 1. Add to aethereal_nexus.py:
#    from nexus_integration import NexusIntegration
#    nexus_integration = NexusIntegration()
#
# 2. In handle_request method:
#    response = await nexus_integration.route_request(request, user_id, ai_name)
#    if response and "error" not in response:
#        return response