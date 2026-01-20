#!/usr/bin/env python3
"""
Enhanced Gabriel's Horn Network Manager
Manages the network of LLM stations with advanced monitoring and analytics
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from llm_station_enhanced import LLMStation
from llm_manager import LLMManager
from enhanced_loki_observer import loki, loki_monitored

class HornNetworkManager:
    """
    Enhanced Gabriel's Horn Network Manager
    Manages a network of LLM-powered stations with advanced routing and monitoring
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.stations = {}  # station_id -> LLMStation
        self.horns = []  # List of horn IDs (special routing stations)
        self.pods = []  # List of pod IDs (endpoint stations)
        self.llm_manager = llm_manager or LLMManager()
        
        # Network topology
        self.rings = {}  # ring_id -> list of station_ids
        self.dimensions = {}  # dimension_id -> list of station_ids
        
        # Network metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "routing_errors": 0,
            "avg_hops": 0,
            "avg_latency_ms": 0
        }
        
        # Log network creation
        loki.log_event(
            {"component": "network_manager", "action": "create"},
            "Gabriel's Horn Network Manager initialized"
        )
    
    def add_station(self, 
                   station_id: str, 
                   station_value: int, 
                   station_type: str = "pod", 
                   model: str = "gemma-2b",
                   dimension: str = "primary") -> LLMStation:
        """
        Add a station to the network
        
        Args:
            station_id: Unique ID for the station
            station_value: Numerical value for routing
            station_type: Type of station (horn or pod)
            model: LLM model to use
            dimension: Network dimension
            
        Returns:
            Created station
        """
        station = LLMStation(station_id, station_value, model, self.llm_manager)
        self.stations[station_id] = station
        
        if station_type == "horn":
            self.horns.append(station_id)
        else:
            self.pods.append(station_id)
        
        # Add to dimension
        if dimension not in self.dimensions:
            self.dimensions[dimension] = []
        self.dimensions[dimension].append(station_id)
        
        loki.log_event(
            {
                "component": "network_manager", 
                "action": "add_station",
                "station_type": station_type,
                "dimension": dimension
            },
            f"Added {station_type} station {station_id} with value {station_value} using model {model}"
        )
        
        return station
    
    def connect_stations(self, station1_id: str, station2_id: str) -> bool:
        """
        Connect two stations
        
        Args:
            station1_id: First station ID
            station2_id: Second station ID
            
        Returns:
            Success status
        """
        if station1_id not in self.stations or station2_id not in self.stations:
            loki.log_event(
                {"component": "network_manager", "action": "error", "error_type": "connection"},
                f"Cannot connect: station not found ({station1_id} or {station2_id})",
                level="error"
            )
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
    
    def create_ring(self, station_ids: List[str], ring_id: Optional[str] = None) -> str:
        """
        Connect stations in a ring topology
        
        Args:
            station_ids: List of station IDs to connect in a ring
            ring_id: Optional ring identifier
            
        Returns:
            Ring ID
        """
        if len(station_ids) < 2:
            loki.log_event(
                {"component": "network_manager", "action": "error", "error_type": "ring_creation"},
                f"Cannot create ring: need at least 2 stations",
                level="error"
            )
            return None
        
        # Generate ring ID if not provided
        if not ring_id:
            ring_id = f"ring-{uuid.uuid4().hex[:8]}"
        
        # Connect stations in a ring
        for i in range(len(station_ids)):
            next_idx = (i + 1) % len(station_ids)
            self.connect_stations(station_ids[i], station_ids[next_idx])
        
        # Store ring
        self.rings[ring_id] = station_ids
        
        loki.log_event(
            {"component": "network_manager", "action": "create_ring", "ring_id": ring_id},
            f"Created ring {ring_id} with stations: {station_ids}"
        )
        
        return ring_id
    
    def create_mesh(self, station_ids: List[str]) -> int:
        """
        Connect stations in a full mesh topology
        
        Args:
            station_ids: List of station IDs to connect in a mesh
            
        Returns:
            Number of connections created
        """
        connections = 0
        
        # Connect each station to every other station
        for i in range(len(station_ids)):
            for j in range(i + 1, len(station_ids)):
                if self.connect_stations(station_ids[i], station_ids[j]):
                    connections += 1
        
        loki.log_event(
            {"component": "network_manager", "action": "create_mesh"},
            f"Created mesh with {len(station_ids)} stations and {connections} connections"
        )
        
        return connections
    
    @loki_monitored(loki, "network_manager")
    async def send_message(self, 
                          source_id: str, 
                          destination_value: int, 
                          content: Any, 
                          priority: str = "normal",
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a message from source to a station with destination value
        
        Args:
            source_id: Source station ID
            destination_value: Destination value
            content: Message content
            priority: Message priority (low, normal, high, critical)
            metadata: Additional message metadata
            
        Returns:
            Message routing result
        """
        if source_id not in self.stations:
            return {"error": "Source station not found"}
        
        # Generate message ID
        message_id = f"msg-{uuid.uuid4().hex}"
        
        # Create message
        message = {
            "id": message_id,
            "content": content,
            "source_id": source_id,
            "source_value": self.stations[source_id].station_value,
            "destination_value": destination_value,
            "priority": priority,
            "timestamp": time.time(),
            "hops": 0,
            "path": [source_id]
        }
        
        # Add metadata if provided
        if metadata:
            message["metadata"] = metadata
        
        self.metrics["messages_sent"] += 1
        
        loki.log_event(
            {
                "component": "network_manager", 
                "action": "send",
                "message_id": message_id,
                "source": source_id,
                "priority": priority
            },
            f"Sending message from {source_id} to value {destination_value}"
        )
        
        # Start routing from source
        start_time = time.time()
        result = await self._route_message(message, source_id)
        
        # Update metrics
        if result.get("status") == "delivered":
            self.metrics["messages_delivered"] += 1
            hops = result.get("hops", 0)
            self.metrics["avg_hops"] = (
                (self.metrics["avg_hops"] * (self.metrics["messages_delivered"] - 1) + hops) / 
                self.metrics["messages_delivered"]
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            self.metrics["avg_latency_ms"] = (
                (self.metrics["avg_latency_ms"] * (self.metrics["messages_delivered"] - 1) + latency_ms) / 
                self.metrics["messages_delivered"]
            )
            
            # Add latency to result
            result["latency_ms"] = latency_ms
        else:
            self.metrics["routing_errors"] += 1
        
        return result
    
    async def _route_message(self, 
                            message: Dict[str, Any], 
                            current_station_id: str, 
                            max_hops: int = 10) -> Dict[str, Any]:
        """
        Route a message through the network
        
        Args:
            message: Message to route
            current_station_id: Current station ID
            max_hops: Maximum number of hops
            
        Returns:
            Routing result
        """
        if message["hops"] >= max_hops:
            loki.log_event(
                {
                    "component": "network_manager", 
                    "action": "max_hops",
                    "message_id": message.get("id", "unknown"),
                    "hops": message["hops"]
                },
                f"Message exceeded max hops: {json.dumps(message)}",
                level="warn"
            )
            return {"error": "Max hops exceeded", "path": message["path"]}
        
        # Process at current station
        station = self.stations[current_station_id]
        result = await station.process_incoming(message)
        
        # Check if delivered
        if result.get("status") == "delivered":
            loki.log_event(
                {
                    "component": "network_manager", 
                    "action": "delivered",
                    "message_id": message.get("id", "unknown"),
                    "destination": current_station_id,
                    "hops": message["hops"]
                },
                f"Message delivered to {current_station_id}"
            )
            return {
                "status": "delivered",
                "destination": current_station_id,
                "path": message["path"],
                "hops": message["hops"],
                "response": result.get("response")
            }
        
        # Continue routing
        next_station_id = result.get("to")
        if not next_station_id or next_station_id not in self.stations:
            loki.log_event(
                {
                    "component": "network_manager", 
                    "action": "no_route",
                    "message_id": message.get("id", "unknown"),
                    "station": current_station_id
                },
                f"No route from {current_station_id}",
                level="warn"
            )
            return {"error": "No route available", "path": message["path"]}
        
        # Update message for next hop
        message["hops"] += 1
        message["path"].append(next_station_id)
        
        # Route to next station
        return await self._route_message(message, next_station_id, max_hops)
    
    @loki_monitored(loki, "network_manager")
    async def broadcast_message(self, 
                              source_id: str, 
                              content: Any, 
                              target_type: str = "all",
                              priority: str = "normal") -> Dict[str, Any]:
        """
        Broadcast a message to multiple stations
        
        Args:
            source_id: Source station ID
            content: Message content
            target_type: Target type (all, horns, pods)
            priority: Message priority
            
        Returns:
            Broadcast results
        """
        if source_id not in self.stations:
            return {"error": "Source station not found"}
        
        # Determine target stations
        if target_type == "horns":
            targets = self.horns
        elif target_type == "pods":
            targets = self.pods
        else:
            targets = list(self.stations.keys())
        
        # Remove source from targets
        if source_id in targets:
            targets.remove(source_id)
        
        # Generate message ID
        message_id = f"broadcast-{uuid.uuid4().hex}"
        
        # Log broadcast
        loki.log_event(
            {
                "component": "network_manager", 
                "action": "broadcast",
                "message_id": message_id,
                "source": source_id,
                "target_type": target_type,
                "targets": len(targets)
            },
            f"Broadcasting message from {source_id} to {len(targets)} {target_type} stations"
        )
        
        # Send to all targets
        results = []
        for target_id in targets:
            target_value = self.stations[target_id].station_value
            result = await self.send_message(
                source_id, 
                target_value, 
                content, 
                priority,
                {"broadcast_id": message_id, "target_type": target_type}
            )
            results.append({
                "target": target_id,
                "result": result
            })
        
        # Summarize results
        delivered = sum(1 for r in results if r["result"].get("status") == "delivered")
        failed = len(results) - delivered
        
        return {
            "broadcast_id": message_id,
            "source": source_id,
            "target_type": target_type,
            "total_targets": len(targets),
            "delivered": delivered,
            "failed": failed,
            "results": results
        }
    
    @loki_monitored(loki, "network_manager")
    async def get_network_status(self) -> Dict[str, Any]:
        """
        Get status of the entire network
        
        Returns:
            Network status information
        """
        # Get status of each station
        station_statuses = {}
        for station_id, station in self.stations.items():
            station_statuses[station_id] = await station.get_status()
        
        # Get network topology
        topology = {
            "stations": len(self.stations),
            "horns": len(self.horns),
            "pods": len(self.pods),
            "rings": {ring_id: stations for ring_id, stations in self.rings.items()},
            "dimensions": {dim: stations for dim, stations in self.dimensions.items()}
        }
        
        # Get connection graph
        connections = []
        for station_id, station in self.stations.items():
            for conn in station.connections:
                connections.append({
                    "from": station_id,
                    "to": conn["id"],
                    "from_value": station.station_value,
                    "to_value": conn["value"]
                })
        
        # Get network metrics from Loki
        sent_query = '{component="network_manager", action="send"}'
        sent_results = loki.query_logs(sent_query, limit=1000)
        sent_count = len(sent_results.get("data", {}).get("result", []))
        
        delivered_query = '{component="network_manager", action="delivered"}'
        delivered_results = loki.query_logs(delivered_query, limit=1000)
        delivered_count = len(delivered_results.get("data", {}).get("result", []))
        
        error_query = '{component="network_manager", action=~"no_route|max_hops"}'
        error_results = loki.query_logs(error_query, limit=1000)
        error_count = len(error_results.get("data", {}).get("result", []))
        
        # Calculate delivery rate
        delivery_rate = delivered_count / sent_count if sent_count > 0 else 0
        
        return {
            "topology": topology,
            "connections": connections,
            "metrics": self.metrics,
            "loki_metrics": {
                "messages_sent": sent_count,
                "messages_delivered": delivered_count,
                "routing_errors": error_count,
                "delivery_rate": delivery_rate
            },
            "station_statuses": station_statuses
        }
    
    @loki_monitored(loki, "network_manager")
    async def analyze_path(self, source_id: str, destination_value: int) -> Dict[str, Any]:
        """
        Analyze potential paths from source to destination
        
        Args:
            source_id: Source station ID
            destination_value: Destination value
            
        Returns:
            Path analysis
        """
        if source_id not in self.stations:
            return {"error": "Source station not found"}
        
        # Find potential destination stations
        potential_destinations = []
        for station_id, station in self.stations.items():
            if station.station_value == destination_value:
                potential_destinations.append(station_id)
        
        if not potential_destinations:
            return {"error": "No stations with the specified destination value"}
        
        # Find shortest paths
        paths = []
        for dest_id in potential_destinations:
            path = await self._find_shortest_path(source_id, dest_id)
            if path:
                paths.append({
                    "destination": dest_id,
                    "path": path,
                    "hops": len(path) - 1
                })
        
        # Sort by hop count
        paths.sort(key=lambda p: p["hops"])
        
        return {
            "source": source_id,
            "destination_value": destination_value,
            "potential_destinations": potential_destinations,
            "paths": paths
        }
    
    async def _find_shortest_path(self, start: str, end: str) -> List[str]:
        """Find shortest path between two stations using BFS"""
        if start not in self.stations or end not in self.stations:
            return []
        
        # BFS
        queue = [[start]]
        visited = {start}
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node == end:
                return path
            
            # Get connections
            station = self.stations[node]
            for conn in station.connections:
                neighbor = conn["id"]
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        
        return []  # No path found

# Example usage
async def main():
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Create network manager
    network = HornNetworkManager(llm_manager)
    
    # Add stations
    print("Adding stations...")
    network.add_station("horn1", 100, "horn", "gemma-2b", "primary")
    network.add_station("horn2", 200, "horn", "hermes-2-pro-llama-3-7b", "primary")
    network.add_station("horn3", 300, "horn", "qwen2.5-14b", "primary")
    network.add_station("pod1", 150, "pod", "gemma-2b", "primary")
    network.add_station("pod2", 250, "pod", "hermes-2-pro-llama-3-7b", "primary")
    network.add_station("pod3", 350, "pod", "qwen2.5-14b", "primary")
    
    # Create connections
    print("Creating connections...")
    network.connect_stations("horn1", "pod1")
    network.connect_stations("horn1", "horn2")
    network.connect_stations("horn2", "pod2")
    network.connect_stations("horn2", "horn3")
    network.connect_stations("horn3", "pod3")
    
    # Create horn ring
    print("Creating horn ring...")
    network.create_ring(["horn1", "horn2", "horn3"], "main-ring")
    
    # Send a message
    print("Sending message...")
    result = await network.send_message("pod1", 350, "Test message", "high")
    print(f"Message result: {result}")
    
    # Broadcast a message
    print("Broadcasting message...")
    broadcast_result = await network.broadcast_message("horn1", "Broadcast test", "pods")
    print(f"Broadcast result: {broadcast_result}")
    
    # Analyze path
    print("Analyzing path...")
    path_analysis = await network.analyze_path("pod1", 350)
    print(f"Path analysis: {path_analysis}")
    
    # Get network status
    print("Getting network status...")
    status = await network.get_network_status()
    print(f"Network topology: {status['topology']}")
    print(f"Network metrics: {status['metrics']}")
    print(f"Loki metrics: {status['loki_metrics']}")

if __name__ == "__main__":
    asyncio.run(main())