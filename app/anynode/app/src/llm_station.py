#!/usr/bin/env python3
"""
LLM-powered Smart Station with Loki integration
"""
import asyncio
import json
from typing import Dict, Any, List
from loki_integration import loki, loki_monitored

class LLMStation:
    def __init__(self, station_id: str, station_value: int, model_name: str = "gemma-2b"):
        self.station_id = station_id
        self.station_value = station_value
        self.model_name = model_name
        self.connections = []
        self.system_prompt = f"""You are station {station_id} with value {station_value}.
Your job is to route messages to their destination efficiently.
Consider numerical proximity, network topology, and message priority."""
    
    def connect(self, station_id: str, station_value: int):
        """Connect to another station"""
        self.connections.append({"id": station_id, "value": station_value})
        loki.log_event(
            {"station": self.station_id, "action": "connect"},
            f"Connected to station {station_id} with value {station_value}"
        )
    
    @loki_monitored
    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route a message using the LLM"""
        # Get routing history from Loki for context
        query = f'{{station="{self.station_id}", action="route"}}'
        history = loki.query_logs(query, limit=5)
        
        # Prepare prompt for the LLM
        prompt = f"""
Message: {json.dumps(message)}
Connected stations: {json.dumps(self.connections)}
Recent routing history: {json.dumps(history)}

Which station should I route this message to? Respond with just the station ID.
"""
        
        # In a real implementation, this would call the LLM
        # For now, simulate the LLM's decision with basic logic
        next_station = self._get_routing_decision(message)
        
        return {
            "action": "route",
            "from": self.station_id,
            "to": next_station,
            "message": message
        }
    
    def _get_routing_decision(self, message: Dict[str, Any]) -> str:
        """Get routing decision based on numerical proximity"""
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
    
    @loki_monitored
    async def process_incoming(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message"""
        # Check if this is the final destination
        if message.get("destination_value") == self.station_value:
            loki.log_event(
                {"station": self.station_id, "action": "deliver"},
                f"Message delivered: {json.dumps(message)}"
            )
            return {"status": "delivered", "station": self.station_id}
        
        # Otherwise, route to next station
        return await self.route_message(message)
    
    @loki_monitored
    async def get_status(self) -> Dict[str, Any]:
        """Get station status"""
        # Query Loki for station metrics
        messages_received = loki.query_logs(f'{{station="{self.station_id}", action="call", function="process_incoming"}}')
        messages_routed = loki.query_logs(f'{{station="{self.station_id}", action="call", function="route_message"}}')
        
        return {
            "station_id": self.station_id,
            "value": self.station_value,
            "model": self.model_name,
            "connections": len(self.connections),
            "messages_received": len(messages_received.get("data", {}).get("result", [])),
            "messages_routed": len(messages_routed.get("data", {}).get("result", []))
        }

# Example usage
async def main():
    # Create stations
    station1 = LLMStation("station1", 100)
    station2 = LLMStation("station2", 200)
    station3 = LLMStation("station3", 300)
    
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
    
    result = await station1.process_incoming(message)
    print(f"Routing result: {result}")
    
    # Get station status
    status = await station1.get_status()
    print(f"Station status: {status}")

if __name__ == "__main__":
    asyncio.run(main())