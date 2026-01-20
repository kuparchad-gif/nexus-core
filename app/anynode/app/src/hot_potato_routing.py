#!/usr/bin/env python3
"""
Hot Potato Routing for Gabriel's Horn Network
Each horn, pod, and token has a numerical ID
Tokens are passed to get closer to destination
"""
import asyncio
import json
from typing import Dict, List, Any

class HotPotatoRouter:
    def __init__(self):
        self.horns = {}  # horn_id -> numerical_value
        self.pods = {}   # pod_id -> numerical_value
        self.connections = {}  # node_id -> list of connected node_ids
    
    def add_horn(self, horn_id: str, value: int):
        """Add a horn with numerical value"""
        self.horns[horn_id] = value
        self.connections[horn_id] = []
    
    def add_pod(self, pod_id: str, value: int):
        """Add a pod with numerical value"""
        self.pods[pod_id] = value
        self.connections[pod_id] = []
    
    def connect(self, node1_id: str, node2_id: str):
        """Connect two nodes"""
        if node1_id in self.connections and node2_id in self.connections:
            if node2_id not in self.connections[node1_id]:
                self.connections[node1_id].append(node2_id)
            if node1_id not in self.connections[node2_id]:
                self.connections[node2_id].append(node1_id)
    
    def get_node_value(self, node_id: str) -> int:
        """Get numerical value of a node (horn or pod)"""
        if node_id in self.horns:
            return self.horns[node_id]
        if node_id in self.pods:
            return self.pods[node_id]
        return None
    
    def find_next_hop(self, current_node: str, destination_value: int) -> str:
        """Find next hop that gets closer to destination value"""
        current_value = self.get_node_value(current_node)
        current_distance = abs(current_value - destination_value)
        
        best_next_hop = None
        best_distance = current_distance
        
        # Check all connected nodes
        for neighbor in self.connections.get(current_node, []):
            neighbor_value = self.get_node_value(neighbor)
            neighbor_distance = abs(neighbor_value - destination_value)
            
            # If this neighbor is closer to destination
            if neighbor_distance < best_distance:
                best_distance = neighbor_distance
                best_next_hop = neighbor
        
        return best_next_hop
    
    async def route_token(self, source_pod: str, destination_value: int, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route a token from source to destination"""
        token = {
            "source": source_pod,
            "source_value": self.get_node_value(source_pod),
            "destination_value": destination_value,
            "message": message,
            "path": [source_pod]
        }
        
        current_node = source_pod
        max_hops = 10  # Prevent infinite loops
        
        for _ in range(max_hops):
            # Find next hop
            next_hop = self.find_next_hop(current_node, destination_value)
            
            if not next_hop:
                return {"error": "No route to destination", "path": token["path"]}
            
            # Add to path
            token["path"].append(next_hop)
            current_node = next_hop
            
            # Check if we've reached a pod with the destination value
            current_value = self.get_node_value(current_node)
            if current_value == destination_value:
                # We've reached destination
                return {
                    "status": "delivered",
                    "destination": current_node,
                    "path": token["path"],
                    "hops": len(token["path"]) - 1,
                    "response": f"Message from {source_pod} delivered to {current_node}"
                }
        
        return {"error": "Max hops exceeded", "path": token["path"]}

# Example usage
async def main():
    router = HotPotatoRouter()
    
    # Add horns
    router.add_horn("horn1", 100)
    router.add_horn("horn2", 200)
    router.add_horn("horn3", 300)
    
    # Add pods
    router.add_pod("pod1", 150)
    router.add_pod("pod2", 250)
    router.add_pod("pod3", 350)
    
    # Connect nodes
    router.connect("horn1", "pod1")
    router.connect("horn1", "horn2")
    router.connect("horn2", "pod2")
    router.connect("horn2", "horn3")
    router.connect("horn3", "pod3")
    
    # Route a token
    message = {"content": "Test message"}
    result = await router.route_token("pod1", 350, message)
    print(f"Routing result: {result}")

if __name__ == "__main__":
    asyncio.run(main())