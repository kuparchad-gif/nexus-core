#!/usr/bin/env python3
"""
Gabriel's Horn Network Manager
Manages the network of LLM stations with Loki monitoring
"""
import asyncio
import json
import random
from typing import Dict, Any, List
from llm_station import LLMStation
from loki_integration import loki, loki_monitored

class HornNetworkManager:
    def __init__(self):
        self.stations = {}  # station_id -> LLMStation
        self.horns = []  # List of horn IDs (special routing stations)
        self.pods = []  # List of pod IDs (endpoint stations)
    
    def add_station(self, station_id: str, station_value: int, station_type: str = "pod", model: str = "gemma-2b"):
        """Add a station to the network"""
        station = LLMStation(station_id, station_value, model)
        self.stations[station_id] = station
        
        if station_type == "horn":
            self.horns.append(station_id)
        else:
            self.pods.append(station_id)
        
        loki.log_event(
            {"component": "network_manager", "action": "add_station"},
            f"Added {station_type} station {station_id} with value {station_value}"
        )
        
        return station
    
    def connect_stations(self, station1_id: str, station2_id: str):
        """Connect two stations"""
        if station1_id not in self.stations or station2_id not in self.stations:
            return False
        
        station1 = self.stations[station1_id]
        station2 = self.stations[station2_id]
        
        station1.connect(station2_id, station2.station_value)
        station2.connect(station1_id, station1.station_value)
        
        loki.log_event(
            {"component": "network_manager", "action": "connect"},
            f"Connected stations {station1_id} and {station2_id}"
        )
        
        return True
    
    def create_ring(self, station_ids: List[str]):
        """Connect stations in a ring topology"""
        if len(station_ids) < 2:
            return False
        
        for i in range(len(station_ids)):
            next_idx = (i + 1) % len(station_ids)
            self.connect_stations(station_ids[i], station_ids[next_idx])
        
        loki.log_event(
            {"component": "network_manager", "action": "create_ring"},
            f"Created ring with stations: {station_ids}"
        )
        
        return True
    
    @loki_monitored
    async def send_message(self, source_id: str, destination_value: int, content: Any, priority: str = "normal") -> Dict[str, Any]:
        """Send a message from source to a station with destination value"""
        if source_id not in self.stations:
            return {"error": "Source station not found"}
        
        message = {
            "content": content,
            "source_id": source_id,
            "source_value": self.stations[source_id].station_value,
            "destination_value": destination_value,
            "priority": priority,
            "timestamp": asyncio.get_event_loop().time(),
            "hops": 0,
            "path": [source_id]
        }
        
        loki.log_event(
            {"component": "network_manager", "action": "send"},
            f"Sending message from {source_id} to value {destination_value}"
        )
        
        # Start routing from source
        return await self._route_message(message, source_id)
    
    async def _route_message(self, message: Dict[str, Any], current_station_id: str, max_hops: int = 10) -> Dict[str, Any]:
        """Route a message through the network"""
        if message["hops"] >= max_hops:
            loki.log_event(
                {"component": "network_manager", "action": "max_hops"},
                f"Message exceeded max hops: {json.dumps(message)}"
            )
            return {"error": "Max hops exceeded", "path": message["path"]}
        
        # Process at current station
        station = self.stations[current_station_id]
        result = await station.process_incoming(message)
        
        # Check if delivered
        if result.get("status") == "delivered":
            loki.log_event(
                {"component": "network_manager", "action": "delivered"},
                f"Message delivered to {current_station_id}"
            )
            return {
                "status": "delivered",
                "destination": current_station_id,
                "path": message["path"],
                "hops": message["hops"]
            }
        
        # Continue routing
        next_station_id = result.get("to")
        if not next_station_id or next_station_id not in self.stations:
            loki.log_event(
                {"component": "network_manager", "action": "no_route"},
                f"No route from {current_station_id}"
            )
            return {"error": "No route available", "path": message["path"]}
        
        # Update message for next hop
        message["hops"] += 1
        message["path"].append(next_station_id)
        
        # Route to next station
        return await self._route_message(message, next_station_id, max_hops)
    
    @loki_monitored
    async def get_network_status(self) -> Dict[str, Any]:
        """Get status of the entire network"""
        station_statuses = {}
        
        for station_id, station in self.stations.items():
            station_statuses[station_id] = await station.get_status()
        
        # Get network metrics from Loki
        messages_sent = loki.query_logs('{component="network_manager", action="send"}')
        messages_delivered = loki.query_logs('{component="network_manager", action="delivered"}')
        
        return {
            "stations": len(self.stations),
            "horns": len(self.horns),
            "pods": len(self.pods),
            "messages_sent": len(messages_sent.get("data", {}).get("result", [])),
            "messages_delivered": len(messages_delivered.get("data", {}).get("result", [])),
            "station_statuses": station_statuses
        }

# Example usage
async def main():
    # Create network manager
    network = HornNetworkManager()
    
    # Add stations
    network.add_station("horn1", 100, "horn", "gemma-2b")
    network.add_station("horn2", 200, "horn", "hermes-2-pro-llama-3-7b")
    network.add_station("horn3", 300, "horn", "qwen2.5-14b")
    network.add_station("pod1", 150, "pod", "gemma-2b")
    network.add_station("pod2", 250, "pod", "hermes-2-pro-llama-3-7b")
    network.add_station("pod3", 350, "pod", "qwen2.5-14b")
    
    # Create connections
    network.connect_stations("horn1", "pod1")
    network.connect_stations("horn1", "horn2")
    network.connect_stations("horn2", "pod2")
    network.connect_stations("horn2", "horn3")
    network.connect_stations("horn3", "pod3")
    
    # Create a ring
    network.create_ring(["horn1", "horn2", "horn3"])
    
    # Send a message
    result = await network.send_message("pod1", 350, "Test message", "high")
    print(f"Message result: {result}")
    
    # Get network status
    status = await network.get_network_status()
    print(f"Network status: {status}")

if __name__ == "__main__":
    asyncio.run(main())