#!/usr/bin/env python3
"""
Hybrid Router for Gabriel's Horn Network
Combines numerical proximity, token ring, and shortest path routing
"""
import asyncio
import heapq
from typing import Dict, List, Any, Tuple

class HybridRouter:
    def __init__(self):
        self.nodes = {}  # node_id -> {value: int, type: str}
        self.connections = {}  # node_id -> list of connected node_ids
        self.rings = {}  # ring_id -> list of node_ids in ring
    
    def add_node(self, node_id: str, value: int, node_type: str = "pod"):
        """Add a node with numerical value"""
        self.nodes[node_id] = {"value": value, "type": node_type}
        self.connections[node_id] = []
    
    def connect(self, node1_id: str, node2_id: str):
        """Connect two nodes"""
        if node1_id in self.connections and node2_id in self.connections:
            if node2_id not in self.connections[node1_id]:
                self.connections[node1_id].append(node2_id)
            if node1_id not in self.connections[node2_id]:
                self.connections[node2_id].append(node1_id)
    
    def create_ring(self, ring_id: str, node_ids: List[str]):
        """Create a token ring with specified nodes"""
        self.rings[ring_id] = node_ids
        # Connect nodes in ring
        for i in range(len(node_ids)):
            self.connect(node_ids[i], node_ids[(i+1) % len(node_ids)])
    
    def _numerical_distance(self, value1: int, value2: int) -> int:
        """Calculate numerical distance between values"""
        return abs(value1 - value2)
    
    def _find_ring_for_node(self, node_id: str) -> str:
        """Find which ring a node belongs to"""
        for ring_id, nodes in self.rings.items():
            if node_id in nodes:
                return ring_id
        return None
    
    def _shortest_path(self, start: str, end: str) -> List[str]:
        """Find shortest path using Dijkstra's algorithm"""
        if start not in self.nodes or end not in self.nodes:
            return []
        
        # Initialize
        distances = {node: float('infinity') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]  # (distance, node)
        previous = {node: None for node in self.nodes}
        
        # Dijkstra's algorithm
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current == end:
                break
                
            if current_distance > distances[current]:
                continue
                
            for neighbor in self.connections.get(current, []):
                distance = current_distance + 1  # All edges have weight 1
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current:
            path.append(current)
            current = previous[current]
        
        return path[::-1]  # Reverse to get start->end
    
    async def route(self, source: str, destination_value: int) -> Dict[str, Any]:
        """Route using hybrid approach"""
        if source not in self.nodes:
            return {"error": "Source node not found"}
        
        # Step 1: Find node with closest value to destination
        closest_node = None
        min_distance = float('infinity')
        
        for node_id, node_data in self.nodes.items():
            if node_data["type"] == "pod":  # Only consider pods as destinations
                distance = self._numerical_distance(node_data["value"], destination_value)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node_id
        
        if not closest_node:
            return {"error": "No suitable destination found"}
        
        # Step 2: Check if source and destination are in same ring
        source_ring = self._find_ring_for_node(source)
        dest_ring = self._find_ring_for_node(closest_node)
        
        if source_ring and source_ring == dest_ring:
            # Use token ring routing (faster within same ring)
            ring_nodes = self.rings[source_ring]
            source_idx = ring_nodes.index(source)
            dest_idx = ring_nodes.index(closest_node)
            
            # Build path around ring
            path = []
            current_idx = source_idx
            while current_idx != dest_idx:
                path.append(ring_nodes[current_idx])
                current_idx = (current_idx + 1) % len(ring_nodes)
            path.append(closest_node)
            
            return {
                "status": "delivered",
                "destination": closest_node,
                "destination_value": self.nodes[closest_node]["value"],
                "path": path,
                "routing": "token_ring",
                "hops": len(path) - 1
            }
        else:
            # Use shortest path routing between rings
            path = self._shortest_path(source, closest_node)
            
            return {
                "status": "delivered" if path else "failed",
                "destination": closest_node,
                "destination_value": self.nodes[closest_node]["value"],
                "path": path,
                "routing": "shortest_path",
                "hops": len(path) - 1 if path else 0
            }

# Example usage
async def main():
    router = HybridRouter()
    
    # Add nodes
    router.add_node("pod1", 100, "pod")
    router.add_node("pod2", 200, "pod")
    router.add_node("pod3", 300, "pod")
    router.add_node("horn1", 150, "horn")
    router.add_node("horn2", 250, "horn")
    router.add_node("horn3", 350, "horn")
    
    # Create connections
    router.connect("pod1", "horn1")
    router.connect("horn1", "pod2")
    router.connect("pod2", "horn2")
    router.connect("horn2", "pod3")
    router.connect("pod3", "horn3")
    router.connect("horn3", "pod1")
    
    # Create rings
    router.create_ring("ring1", ["pod1", "horn1", "pod2"])
    router.create_ring("ring2", ["pod2", "horn2", "pod3", "horn3"])
    
    # Test routing
    result1 = await router.route("pod1", 300)
    print(f"Route from pod1 to value 300: {result1}")
    
    result2 = await router.route("pod1", 200)
    print(f"Route from pod1 to value 200: {result2}")

if __name__ == "__main__":
    asyncio.run(main())