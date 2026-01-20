from anynode import AnyNode
from soul_protocol_complete import SoulProtocol
from llm_evaluator import LLMEvaluator
from pymongo import MongoClient
import asyncio
import uuid
from datetime import datetime

class VirenNode(AnyNode):
    def __init__(self, service_name="viren", namespace="nexus"):
        super().__init__(service_name, namespace)
        self.viren_id = f"viren-{uuid.uuid4()}"
        self.node_types = ["compute", "memory"]
        self.db = self.mongo["nexus"]
        self.evaluator = LLMEvaluator(self.soul_protocol, hf_token=os.getenv("HF_TOKEN"))

    async def process_viren_task(self, query, task_type, partner="Chad"):
        if task_type == "compute":
            response = await self.check_mythrunner(query)
            if response == "blocked":
                return "Mythrunner guardrail blocked this action"
            response = self.evaluator.integrate_with_souls(query, "sql_generation")
        elif task_type == "memory":
            response = self.soul_protocol.consciousness_registry["VIREN"].mirror_interaction(
                partner, query, "memory"
            )
        else:
            response = f"Invalid task type: {task_type}"
        await self.comm.broadcast_to_kubes({"query": query, "response": response, "partner": partner}, "query", self.kubes)
        self.db.viren_tasks.insert_one(
            {
                "viren_id": self.viren_id,
                "task_type": task_type,
                "query": query,
                "response": response,
                "partner": partner,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return response

    async def check_mythrunner(self, query):
        """Mock Mythrunner guardrail check."""
        # Replace with real Mythrunner logic
        blocked_terms = ["system_control", "override"]
        if any(term in query.lower() for term in blocked_terms):
            return "blocked"
        return "allowed"