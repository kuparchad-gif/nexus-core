from nexus_platform.common.communication import CommunicationLayer
from nexus_platform.common.discovery import ServiceDiscovery
from trumpet_structure import TrumpetStructure
from soul_protocol_complete import SoulProtocol
from llm_evaluator_complete import LLMEvaluator
from pymongo import MongoClient
import uuid
import asyncio
from datetime import datetime

class AnyNode:
    def __init__(self, service_name="anynode", namespace="nexus"):
        self.comm = CommunicationLayer(service_name)
        self.discovery = ServiceDiscovery(service_name)
        self.trumpet = TrumpetStructure()
        self.soul_protocol = SoulProtocol()
        self.evaluator = LLMEvaluator(self.soul_protocol)
        self.mongo = MongoClient("mongodb://localhost:27017")
        self.db = self.mongo[namespace]
        self.kubes = {}
        self.namespace = namespace

    def register_kube(self, kube_id, address, port, kube_type):
        full_address = f"{self.namespace}/{kube_type}/{kube_id}:{port}"
        self.discovery.register_service(kube_id, address, port)
        self.kubes[kube_id] = {
            'address': address,
            'port': port,
            'type': kube_type,
            'full_address': full_address
        }
        self.db.kubes.update_one(
            {'kube_id': kube_id},
            {'$set': {
                'address': address,
                'port': port,
                'type': kube_type,
                'full_address': full_address,
                'registered_at': datetime.now().isoformat()
            }},
            upsert=True
        )
        # Start WebSocket server for kube
        asyncio.run(self.comm.start_websocket_server(address, port))
        self.comm.logger.info({"action": "kube_registered", "kube_id": kube_id, "full_address": full_address})

    def discover_kubes(self):
        services = self.discovery.discover_services("cognikube")
        for service in services:
            kube_id = f"cognikube_{service['address']}:{service['port']}"
            kube_type = "unknown"
            full_address = f"{self.namespace}/{kube_type}/{kube_id}:{service['port']}"
            self.kubes[kube_id] = {
                'address': service['address'],
                'port': service['port'],
                'type': kube_type,
                'full_address': full_address
            }
            self.db.kubes.update_one(
                {'kube_id': kube_id},
                {'$set': self.kubes[kube_id]},
                upsert=True
            )
            asyncio.run(self.comm.start_websocket_server(service['address'], service['port']))
        return self.kubes

    async def relay_to_kubes(self, data, data_type):
        databases = {k: self.comm.qdrant for k in self.kubes}
        self.trumpet.pulse_replication(databases)
        await self.comm.broadcast_to_kubes(data, data_type, self.kubes)
        # Evaluate and process with SQLCoder if needed
        if data_type == "query":
            response = self.evaluator.integrate_with_souls(data.get("query", ""), "sql_generation")
            self.db.souls.update_one(
                {"soul_name": "LILLITH"},
                {"$set": {"last_query": data.get("query"), "response": response, "timestamp": datetime.now().isoformat()}},
                upsert=True
            )
            return response