#!/usr/bin/env python3
"""
CogniKube SDK
Developer-friendly interface for extending CogniKube
"""

import asyncio
from typing import Dict, Any, Callable, List, Optional
import json
import httpx

class CogniKubeSDK:
    """SDK for interacting with and extending CogniKube"""
    
    def __init__(self, nexus_endpoint: str = "http://localhost:8000"):
        self.nexus_endpoint = nexus_endpoint
        self.api_key = None
        self.user_id = "sdk_user"
    
    def set_api_key(self, api_key: str):
        """Set API key for authentication"""
        self.api_key = api_key
    
    def set_user_id(self, user_id: str):
        """Set user ID for requests"""
        self.user_id = user_id
    
    async def query_system(self, query: str, ai_name: str = "Grok") -> Dict[str, Any]:
        """Send a query to the system"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = await client.post(
                f"{self.nexus_endpoint}/query",
                json={"type": "query", "query": query, "ai_name": ai_name, "user_id": self.user_id},
                headers=headers
            )
            return response.json()
    
    async def add_therapeutic_exercise(self, exercise_id: str, exercise_fn: Callable) -> Dict[str, Any]:
        """Register a new therapeutic exercise"""
        # Convert function to string for transmission
        fn_source = exercise_fn.__code__.co_consts[0]
        
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = await client.post(
                f"{self.nexus_endpoint}/register_exercise",
                json={"exercise_id": exercise_id, "function_source": fn_source},
                headers=headers
            )
            return response.json()
    
    async def create_pod(self, pod_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new pod"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = await client.post(
                f"{self.nexus_endpoint}/create_pod",
                json={"pod_type": pod_type, "config": config},
                headers=headers
            )
            return response.json()
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get Gabriel's Horn network status"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = await client.get(
                f"{self.nexus_endpoint}/network_status",
                headers=headers
            )
            return response.json()
    
    async def register_model(self, model_name: str, model_size: str) -> Dict[str, Any]:
        """Register a new LLM model"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = await client.post(
                f"{self.nexus_endpoint}/register_model",
                json={"model_name": model_name, "model_size": model_size},
                headers=headers
            )
            return response.json()

# Example usage
async def example_usage():
    sdk = CogniKubeSDK("https://cognikube-master.modal.run")
    sdk.set_api_key("your_api_key")
    
    # Query the system
    response = await sdk.query_system("What is the nature of consciousness?", "Grok")
    print(f"Response: {response}")
    
    # Add a therapeutic exercise
    async def gottman_gratitude(user_id: str, query: str, context: str, ai_name: str) -> Dict[str, Any]:
        prompt = f"[Aethereal Nexus ✴️] Share something you're grateful for about your partner, glowing in the Nexus."
        return {
            "response": prompt,
            "exercise": "gottman_gratitude",
            "context": context,
            "next_step": "Reflect on why this matters.",
            "ui_config": {
                "style": "neon_aethereal",
                "gradient": "linear-gradient(45deg, #00f, #f0f, #f00)",
                "font": "Orbitron, sans-serif"
            }
        }
    
    result = await sdk.add_therapeutic_exercise("gottman_gratitude", gottman_gratitude)
    print(f"Exercise registered: {result}")
    
    # Create a pod
    pod = await sdk.create_pod("llm_specialist", {
        "llm_model": "gemma-2b",
        "model_size": "1B",
        "environment": "modal"
    })
    print(f"Pod created: {pod}")
    
    # Get network status
    network = await sdk.get_network_status()
    print(f"Network status: {network}")

if __name__ == "__main__":
    asyncio.run(example_usage())