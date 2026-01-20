#!/usr/bin/env python3
"""
LLM Manager for CogniKube
Integrates with Hugging Face Inference API for real LLM capabilities
"""
import os
import json
import asyncio
import httpx
import time
from typing import Dict, Any, List, Optional, Union
import numpy as np
from transformers import AutoTokenizer

class LLMManager:
    """
    Manages LLM interactions for CogniKube
    Supports multiple models and fallback mechanisms
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"
        self.models = {
            "gemma-2b": "google/gemma-2b",
            "hermes-2-pro-llama-3-7b": "NousResearch/hermes-2-pro-llama-3-7b",
            "qwen2.5-14b": "Qwen/Qwen2.5-14B"
        }
        self.tokenizers = {}
        self.cache = {}
        self.last_call = {}
        self.rate_limit_delay = 1.0  # seconds between API calls
        
        # Initialize tokenizers
        self._init_tokenizers()
    
    def _init_tokenizers(self):
        """Initialize tokenizers for supported models"""
        try:
            for model_name, model_id in self.models.items():
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Error initializing tokenizers: {e}")
    
    async def generate(self, 
                      prompt: str, 
                      model: str = "gemma-2b", 
                      max_tokens: int = 256,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate text using the specified model
        
        Args:
            prompt: The input prompt
            model: Model name (gemma-2b, hermes-2-pro-llama-3-7b, qwen2.5-14b)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_cache: Whether to use cache for identical prompts
        
        Returns:
            Dictionary with generated text and metadata
        """
        # Check if result is in cache
        cache_key = f"{model}:{prompt}:{max_tokens}:{temperature}:{top_p}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limiting
        if model in self.last_call:
            elapsed = time.time() - self.last_call[model]
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        
        # Get model ID
        model_id = self.models.get(model)
        if not model_id:
            return {"error": f"Unsupported model: {model}", "text": ""}
        
        # Prepare request
        url = f"{self.base_url}/{model_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False
            }
        }
        
        try:
            # Make API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                self.last_call[model] = time.time()
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Process result
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get("generated_text", "")
                    else:
                        text = result.get("generated_text", "")
                    
                    output = {
                        "text": text,
                        "model": model,
                        "prompt": prompt,
                        "timestamp": time.time()
                    }
                    
                    # Cache result
                    if use_cache:
                        self.cache[cache_key] = output
                    
                    return output
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    print(error_msg)
                    
                    # Try fallback model if available
                    if model != "gemma-2b":
                        print(f"Falling back to gemma-2b")
                        return await self.generate(prompt, "gemma-2b", max_tokens, temperature, top_p, use_cache)
                    
                    return {"error": error_msg, "text": ""}
        
        except Exception as e:
            error_msg = f"Request Error: {str(e)}"
            print(error_msg)
            return {"error": error_msg, "text": ""}
    
    async def get_embedding(self, text: str, model: str = "gemma-2b") -> List[float]:
        """Get embedding vector for text"""
        # Use tokenizer to get embedding (simplified)
        if model in self.tokenizers:
            try:
                tokenizer = self.tokenizers[model]
                tokens = tokenizer(text, return_tensors="np")
                # This is a simplified embedding - in a real implementation, 
                # you would use a proper embedding model
                embedding = np.mean(tokens["input_ids"], axis=1).tolist()[0]
                return embedding
            except Exception as e:
                print(f"Embedding error: {e}")
        
        # Fallback to random embedding
        return [np.random.random() for _ in range(384)]
    
    async def get_routing_decision(self, 
                                  station_id: str,
                                  message: Dict[str, Any],
                                  connections: List[Dict[str, Any]],
                                  routing_history: List[Dict[str, Any]] = None) -> str:
        """
        Get routing decision from LLM
        
        Args:
            station_id: ID of the current station
            message: Message to route
            connections: List of connected stations with their values
            routing_history: Previous routing decisions (optional)
        
        Returns:
            ID of the next station to route to
        """
        # Prepare prompt
        prompt = f"""
You are station {station_id}, a routing node in the Gabriel's Horn network.
Your job is to route messages efficiently to their destination.

Message: {json.dumps(message, indent=2)}

Your connections:
{json.dumps(connections, indent=2)}

"""
        
        if routing_history:
            prompt += f"""
Recent routing history:
{json.dumps(routing_history, indent=2)}

"""
        
        prompt += """
Which station should I route this message to? Respond with just the station ID.
"""
        
        # Get model based on station_id
        model = "gemma-2b"
        if "horn2" in station_id or "pod2" in station_id:
            model = "hermes-2-pro-llama-3-7b"
        elif "horn3" in station_id or "pod3" in station_id:
            model = "qwen2.5-14b"
        
        # Generate response
        result = await self.generate(prompt, model, max_tokens=32, temperature=0.3)
        
        # Extract station ID from response
        response_text = result.get("text", "").strip()
        
        # Check if response matches any connection
        for connection in connections:
            if connection["id"] in response_text:
                return connection["id"]
        
        # Fallback to numerical proximity
        destination_value = message.get("destination_value", 0)
        closest_station = None
        min_distance = float('infinity')
        
        for station in connections:
            distance = abs(station["value"] - destination_value)
            if distance < min_distance:
                min_distance = distance
                closest_station = station["id"]
        
        return closest_station

# Example usage
async def main():
    llm_manager = LLMManager()
    
    # Test generation
    result = await llm_manager.generate(
        "What is the Gabriel's Horn paradox in mathematics?",
        "gemma-2b"
    )
    print(f"Generation result: {result}")
    
    # Test routing decision
    message = {
        "content": "Test message",
        "source_id": "pod1",
        "destination_value": 350,
        "priority": "high"
    }
    
    connections = [
        {"id": "horn1", "value": 100},
        {"id": "pod2", "value": 250},
        {"id": "horn2", "value": 200}
    ]
    
    routing_history = [
        {"from": "pod1", "to": "horn1", "message_id": "msg1"},
        {"from": "horn1", "to": "horn2", "message_id": "msg1"}
    ]
    
    next_station = await llm_manager.get_routing_decision(
        "pod1", message, connections, routing_history
    )
    print(f"Routing decision: {next_station}")

if __name__ == "__main__":
    asyncio.run(main())