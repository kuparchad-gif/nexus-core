#!/usr/bin/env python3
"""
Numerical Token Ring for Gabriel's Horn
Routes tokens to nodes with closest numerical value
"""
import asyncio
import json
from typing import Dict, List, Any

class NumericalRing:
    def __init__(self):
        self.nodes = {}  # node_id -> {value: int, info: Dict}
        self.current_token = None
    
    def add_node(self, node_id: str, numerical_value: int, node_info: Dict[str, Any] = None):
        """Add a node with a numerical value to the ring"""
        self.nodes[node_id] = {
            "value": numerical_value,
            "info": node_info or {}
        }
        print(f"Node {node_id} added with value {numerical_value}")
    
    def find_closest_node(self, target_value: int) -> str:
        """Find the node with the closest numerical value to the target"""
        if not self.nodes:
            return None
        
        closest_node = None
        min_distance = float('inf')
        
        for node_id, node_data in self.nodes.items():
            distance = abs(node_data["value"] - target_value)
            if distance < min_distance:
                min_distance = distance
                closest_node = node_id
        
        return closest_node
    
    async def route_token(self, token_value: int, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route token to the node with closest numerical value"""
        target_node = self.find_closest_node(token_value)
        if not target_node:
            return {"error": "No nodes in ring"}
        
        self.current_token = target_node
        print(f"Token {token_value} routed to node {target_node} with value {self.nodes[target_node]['value']}")
        
        # Process at target node
        response = await self._process_at_node(target_node, message)
        return response
    
    async def _process_at_node(self, node_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message at a specific node"""
        node_data = self.nodes.get(node_id, {})
        node_value = node_data.get("value", 0)
        
        # Simulate processing
        print(f"Processing at node {node_id} with value {node_value}")
        
        # Return response
        return {
            "processed_by": node_id,
            "node_value": node_value,
            "response": f"Processed message: {message.get('content', '')}"
        }

# Example usage
async def main():
    ring = NumericalRing()
    
    # Add nodes with numerical values
    ring.add_node("node1", 100, {"type": "sender"})
    ring.add_node("node2", 200, {"type": "router"})
    ring.add_node("node3", 300, {"type": "receiver"})
    ring.add_node("node4", 150, {"type": "router"})
    
    # Route token with value 160
    message = {"content": "Test message"}
    response = await ring.route_token(160, message)
    print(f"Response: {response}")
    
    # Route another token with different value
    response = await ring.route_token(250, message)
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())