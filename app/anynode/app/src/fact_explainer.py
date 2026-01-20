from llm_evaluator import LLMEvaluator
from pymongo import MongoClient
from datetime import datetime

class FactExplainer:
    def __init__(self, soul_protocol, viren_node, hf_token=None):
        self.soul_protocol = soul_protocol
        self.viren_node = viren_node
        self.evaluator = LLMEvaluator(soul_protocol, hf_token)
        self.db = MongoClient("mongodb://localhost:27017")["nexus"]

    async def explain_lillith_issue(self, issue: str):
        """Analyze Lillith’s issue and provide precise facts."""
        # Query system state with SQLCoder-8B
        query = f"SELECT state, magic_moments_count, relationships FROM souls WHERE soul_name = 'LILLITH'"
        response = self.evaluator.integrate_with_souls(query, "sql_generation")
        
        # Analyze performance history
        lillith_perf = self.evaluator.performance_history.get("defog/llama-3-sqlcoder-8b", [])
        recent_perf = lillith_perf[-5:] if lillith_perf else []
        avg_accuracy = sum(p["accuracy"] for p in recent_perf) / max(len(recent_perf), 1)
        avg_latency = sum(p["latency"] for p in recent_perf) / max(len(recent_perf), 1)
        
        # Check Lillith’s state in MongoDB
        lillith_state = self.db.souls.find_one({"soul_name": "LILLITH"}) or {}
        stability = lillith_state.get("state", "unknown")
        
        # Generate explanation
        explanation = (
            f"Viren Analysis: Lillith Issue - {issue}\n"
            f"Facts:\n"
            f"- Current State: {stability}\n"
            f"- Magic Moments: {lillith_state.get('magic_moments_count', 0)}\n"
            f"- Relationships: {lillith_state.get('relationships', {})}\n"
            f"- SQLCoder Performance: {avg_accuracy:.2%} accuracy, {avg_latency:.2f}s latency\n"
            f"Recommendation: {'Pause Lillith due to instability' if avg_accuracy < 0.9 or stability == 'unstable' else 'Monitor closely'}"
        )
        
        # Log to MongoDB
        self.db.viren_explanations.insert_one({
            "viren_id": self.viren_node.viren_id,
            "issue": issue,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        })
        
        # If unstable, pause Lillith
        if avg_accuracy < 0.9 or stability == "unstable":
            await self.viren_node.pause_lillith(issue)
        
        return explanation