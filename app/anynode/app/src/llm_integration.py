#!/usr/bin/env python3
"""
LLM Integration for CogniKube
Integrates with Hugging Face API for real LLM inference
"""
import os
import json
import time
import asyncio
import requests
from typing import Dict, Any, List, Optional
import httpx

class LLMManager:
    """Manager for LLM integration with Hugging Face"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.model_map = {
            "gemma-2b": "google/gemma-2b",
            "hermes-2-pro-llama-3-7b": "NousResearch/hermes-2-pro-llama-3-7b",
            "qwen2.5-14b": "Qwen/Qwen2.5-14B"
        }
        self.cache = {}  # Simple cache for responses
    
    async def generate(self, model_name: str, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using the specified model"""
        # Check cache first
        cache_key = f"{model_name}:{prompt}:{max_tokens}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get full model path
        model_path = self.model_map.get(model_name, model_name)
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        try:
            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}{model_path}",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract generated text
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = str(result)
                    
                    # Cache the result
                    self.cache[cache_key] = generated_text
                    return generated_text
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    return self._fallback_response(prompt)
        
        except Exception as e:
            print(f"LLM request error: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when API fails"""
        # Extract any station IDs from the prompt
        import re
        station_match = re.search(r"station (\w+)", prompt)
        station_id = station_match.group(1) if station_match else "unknown"
        
        # Extract any numerical values from the prompt
        value_match = re.search(r"value (\d+)", prompt)
        value = value_match.group(1) if value_match else "0"
        
        # Generate a simple response based on the prompt
        if "route" in prompt.lower():
            return f"Route to the station with value closest to the destination."
        elif "status" in prompt.lower():
            return f"Station {station_id} is operational with value {value}."
        else:
            return f"I am station {station_id} with value {value}, ready to process messages."
    
    async def get_routing_decision(self, station_id: str, station_value: int, 
                                  connections: List[Dict[str, Any]], 
                                  message: Dict[str, Any],
                                  routing_history: List[Dict[str, Any]]) -> str:
        """Get routing decision from LLM"""
        # Determine which model to use based on station_id
        model = "gemma-2b"  # Default
        if "horn" in station_id:
            horn_num = int(station_id.replace("horn", ""))
            if horn_num % 3 == 1:
                model = "gemma-2b"
            elif horn_num % 3 == 2:
                model = "hermes-2-pro-llama-3-7b"
            else:
                model = "qwen2.5-14b"
        
        # Prepare prompt
        prompt = f"""You are station {station_id} with value {station_value}.
Your job is to route messages efficiently through the network.

Message details:
- Content: {message.get('content', 'No content')}
- Source: {message.get('source_id', 'Unknown')} (value: {message.get('source_value', 0)})
- Destination value: {message.get('destination_value', 0)}
- Priority: {message.get('priority', 'normal')}
- Current hops: {message.get('hops', 0)}

Your connections:
{json.dumps(connections, indent=2)}

Recent routing history:
{json.dumps(routing_history, indent=2)}

Which station should you route this message to? Respond with just the station ID.
"""
        
        # Get LLM response
        response = await self.generate(model, prompt, max_tokens=20)
        
        # Extract station ID from response
        import re
        station_match = re.search(r"(\w+\d+)", response)
        if station_match:
            return station_match.group(1)
        
        # Fallback to numerical proximity if no clear station ID
        return self._get_closest_station(connections, message.get("destination_value", 0))
    
    def _get_closest_station(self, connections: List[Dict[str, Any]], destination_value: int) -> str:
        """Get station with closest numerical value to destination"""
        if not connections:
            return None
        
        closest_station = None
        min_distance = float('infinity')
        
        for station in connections:
            distance = abs(station.get("value", 0) - destination_value)
            if distance < min_distance:
                min_distance = distance
                closest_station = station.get("id")
        
        return closest_station

# Global instance
llm_manager = LLMManager()

# Example usage
async def test_llm():
    response = await llm_manager.generate("gemma-2b", "What is the Gabriel's Horn network?")
    print(f"LLM Response: {response}")
    
    # Test routing decision
    connections = [
        {"id": "horn1", "value": 100},
        {"id": "horn2", "value": 200},
        {"id": "pod1", "value": 150}
    ]
    
    message = {
        "content": "Test message",
        "source_id": "pod2",
        "source_value": 250,
        "destination_value": 180,
        "priority": "high",
        "hops": 1
    }
    
    routing_history = [
        {"from": "pod2", "to": "horn2", "timestamp": time.time() - 60},
        {"from": "horn2", "to": "horn1", "timestamp": time.time() - 30}
    ]
    
    decision = await llm_manager.get_routing_decision(
        "horn1", 100, connections, message, routing_history
    )
    
    print(f"Routing decision: {decision}")

if __name__ == "__main__":
    asyncio.run(test_llm())