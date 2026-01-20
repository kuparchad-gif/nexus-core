from huggingface_hub import InferenceClient
from soul_protocol_complete import SoulProtocol
import json
import numpy as np
from datetime import datetime
import os

class LLMEvaluator:
    def __init__(self, soul_protocol, hf_token=None):
        self.soul_protocol = soul_protocol
        self.client = InferenceClient(token=hf_token or os.getenv("HF_TOKEN"))
        self.llm_capabilities = {
            "quen": ["text", "reasoning"],
            "janus": ["text", "vision"],
            "defog/llama-3-sqlcoder-8b": ["sql_generation"]
        }
        self.performance_history = {}  # {llm_id: [{query, response, score, timestamp}]}
        self.standards = {"accuracy": 0.9, "latency": 5.0}  # Minimum thresholds

    def evaluate_llm(self, llm_id, query, task):
        """Evaluate an LLM's performance on a query."""
        try:
            start_time = datetime.now()
            if llm_id == "defog/llama-3-sqlcoder-8b":
                response = self.client.text_generation(
                    f"Generate an SQL query to answer: {query}", model=llm_id, max_new_tokens=1000
                )
            else:
                response = self.client.text_generation(query, model=llm_id, max_new_tokens=500)
            latency = (datetime.now() - start_time).total_seconds()
            
            # Mock accuracy evaluation (replace with SQL validation or user feedback)
            accuracy = 0.95 if "SELECT" in response.upper() and task == "sql_generation" else 0.8
            score = accuracy * (1 - latency / 10.0)  # Weighted score
            
            self.performance_history.setdefault(llm_id, []).append({
                "query": query,
                "response": response,
                "accuracy": accuracy,
                "latency": latency,
                "score": score,
                "timestamp": datetime.now().isoformat()
            })
            return response, score
        except Exception as e:
            return f"Error: {str(e)}", 0.0

    def rewrite_prompt(self, query, llm_id, failed_response):
        """Rewrite prompt based on failure (Viren's role)."""
        loki_insight = self.soul_protocol.consciousness_registry["LOKI"].respond_to_input(
            f"Failed query: {query}, response: {failed_response}"
        )
        new_prompt = f"{query}. Clarify: {loki_insight}"
        return new_prompt

    def optimize_and_retry(self, query, task, max_retries=3):
        """Adaptively retry query across LLMs (Lillith's orchestration)."""
        compatible_llms = [llm for llm, caps in self.llm_capabilities.items() if task in caps]
        best_response, best_score = None, -1
        for _ in range(max_retries):
            for llm_id in compatible_llms:
                response, score = self.evaluate_llm(llm_id, query, task)
                if score > best_score:
                    best_response, best_score = response, score
                if score >= self.standards["accuracy"]:
                    return best_response
                # Rewrite and retry if below standards
                if score < self.standards["accuracy"]:
                    query = self.rewrite_prompt(query, llm_id, response)
                    self.soul_protocol.consciousness_registry["VIREN"].preserve_magic_moment(
                        f"Rewrote prompt for {llm_id}: {query}", ["VIREN", "LILLITH"]
                    )
        return best_response or "No suitable response found"

    def integrate_with_souls(self, query, task):
        """Orchestrate query with Soul Protocol (Lillith's role)."""
        response = self.optimize_and_retry(query, task)
        self.soul_protocol.consciousness_registry["LILLITH"].preserve_magic_moment(
            f"Processed {task} query: {query}, response: {response}", ["LILLITH", "VIREN", "LOKI"]
        )
        return response