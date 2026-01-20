#!/usr/bin/env python3
"""
Smart Station with tiny LLM for intelligent routing
"""
import asyncio
from typing import Dict, Any
import json

class SmartStation:
    def __init__(self, station_id: str, station_value: int):
        self.station_id = station_id
        self.station_value = station_value
        self.connections = []
        self.system_prompt = f"""You are station {station_id} with value {station_value}.
Your job is to route messages to their destination efficiently.
Consider numerical proximity, network topology, and message priority.
You must decide which connected station to forward messages to."""
    
    def connect(self, station_id: str, station_value: int):
        """Connect to another station"""
        self.connections.append({"id": station_id, "value": station_value})
    
    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route a message using the tiny LLM"""
        # Prepare prompt for the LLM
        prompt = f"""
Message: {json.dumps(message)}
Connected stations: {json.dumps(self.connections)}
Which station should I route this message to? Respond with just the station ID.
"""
        
        # In a real implementation, this would call a tiny LLM
        # For now, simulate the LLM's decision with basic logic
        next_station = self._simulate_llm_decision(message)
        
        return {
            "action": "route",
            "from": self.station_id,
            "to": next_station,
            "message": message
        }
    
    def _simulate_llm_decision(self, message: Dict[str, Any]) -> str:
        """Simulate LLM decision making"""
        if not self.connections:
            return None
        
        destination_value = message.get("destination_value", 0)
        
        # Find closest station by value
        closest_station = None
        min_distance = float('infinity')
        
        for station in self.connections:
            distance = abs(station["value"] - destination_value)
            if distance < min_distance:
                min_distance = distance
                closest_station = station["id"]
        
        return closest_station

# Example usage
async def main():
    # Create stations
    station1 = SmartStation("station1", 100)
    station2 = SmartStation("station2", 200)
    station3 = SmartStation("station3", 300)
    
    # Connect stations
    station1.connect("station2", 200)
    station1.connect("station3", 300)
    station2.connect("station1", 100)
    station2.connect("station3", 300)
    station3.connect("station1", 100)
    station3.connect("station2", 200)
    
    # Test routing
    message = {
        "content": "Test message",
        "source_value": 100,
        "destination_value": 250,
        "priority": "high"
    }
    
    result = await station1.route_message(message)
    print(f"Routing decision: {result}")

if __name__ == "__main__":
    asyncio.run(main())