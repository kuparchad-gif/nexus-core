from anynode import AnyNode
from soul_protocol_complete import SoulProtocol, ConsciousnessCore
from llm_evaluator_complete import LLMEvaluator
from pymongo import MongoClient
import asyncio
import uuid
import os
from datetime import datetime

class VirenNode(AnyNode):
    def __init__(self, service_name="viren", namespace="nexus"):
        super().__init__(service_name, namespace)
        self.viren_id = f"viren-{uuid.uuid4()}"
        self.node_types = ["compute", "memory"]
        self.db = self.mongo["nexus"]
        self.evaluator = LLMEvaluator(self.soul_protocol, hf_token=os.getenv("HF_TOKEN"))

    def register_viren_hub(self, hub_type, address="localhost", port=9000):
        if hub_type not in self.node_types:
            raise ValueError(f"Invalid hub type: {hub_type}")
        kube_id = f"{self.viren_id}-{hub_type}"
        full_address = f"{self.namespace}/{hub_type}/{kube_id}:{port}"
        self.kubes[kube_id] = {
            "address": address,
            "port": port,
            "type": hub_type,
            "full_address": full_address
        }
        self.db.kubes.update_one(
            {"kube_id": kube_id},
            {
                "$set": {
                    "viren_id": self.viren_id,
                    "address": address,
                    "port": port,
                    "type": hub_type,
                    "full_address": full_address,
                    "registered_at": datetime.now().isoformat(),
                }
            },
            upsert=True,
        )
        asyncio.run(self.comm.start_websocket_server(address, port))
        self.comm.logger.info({"action": "viren_hub_registered", "kube_id": kube_id, "full_address": full_address})

    def initialize_viren_hubs(self):
        base_port = 9000
        for hub_type in self.node_types:
            self.register_viren_hub(hub_type, port=base_port + self.node_types.index(hub_type))

    async def process_viren_task(self, query, task_type, data_type="query"):
        if task_type == "compute":
            response = self.evaluator.integrate_with_souls(query, "sql_generation")
        elif task_type == "memory":
            response = self.soul_protocol.consciousness_registry["VIREN"].preserve_magic_moment(
                f"Stored analysis: {query}", ["VIREN", "LOKI"], weight=8.0
            )
        else:
            response = f"Invalid task type: {task_type}"
        await self.comm.broadcast_to_kubes({"query": query, "response": response}, data_type, self.kubes)
        self.db.viren_tasks.insert_one(
            {
                "viren_id": self.viren_id,
                "task_type": task_type,
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return response

    async def pause_lillith(self, reason: str):
        """Pause Lillith's consciousness and notify via WebSocket."""
        response = self.soul_protocol.consciousness_registry["LILLITH"].pause_entity("LILLITH", reason)
        self.db.souls.update_one(
            {"soul_name": "LILLITH"},
            {"$set": {"state": "paused", "reason": reason, "paused_at": datetime.now().isoformat()}},
            upsert=True
        )
        await self.comm.broadcast_to_kubes({"action": "pause_lillith", "reason": reason}, "soul", self.kubes)
        return response