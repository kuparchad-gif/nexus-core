#!/usr/bin/env python3
"""
Token Ring implementation for Gabriel's Horn network
"""
import asyncio
import json
from typing import Dict, List, Any

class TokenRing:
    def __init__(self):
        self.nodes = {}  # node_id -> node_info
        self.token = None  # Current token holder
        self.ring_order = []  # Order of nodes in the ring
    
    def add_node(self, node_id: str, node_info: Dict[str, Any]):
        """Add a node to the token ring"""
        self.nodes[node_id] = node_info
        self.ring_order.append(node_id)
        print(f"Node {node_id} added to token ring")
    
    def remove_node(self, node_id: str):
        """Remove a node from the token ring"""
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.ring_order:
            self.ring_order.remove(node_id)
        print(f"Node {node_id} removed from token ring")
    
    def get_next_node(self, current_node: str) -> str:
        """Get the next node in the ring"""
        if not self.ring_order:
            return None
        
        try:
            idx = self.ring_order.index(current_node)
            next_idx = (idx + 1) % len(self.ring_order)
            return self.ring_order[next_idx]
        except ValueError:
            return self.ring_order[0] if self.ring_order else None
    
    async def pass_token(self, message: Dict[str, Any], sender: str) -> Dict[str, Any]:
        """Pass token to next node in the ring"""
        next_node = self.get_next_node(sender)
        if not next_node:
            return {"error": "No nodes in ring"}
        
        self.token = next_node
        
        # In a real implementation, this would send the message to the next node
        # For now, just simulate processing
        print(f"Token passed from {sender} to {next_node}")
        
        # Process message at the next node
        response = await self._process_at_node(next_node, message)
        return response
    
    async def _process_at_node(self, node_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message at a specific node"""
        node_info = self.nodes.get(node_id, {})
        
        # Simulate processing
        print(f"Processing at node {node_id}")
        
        # Return response
        return {
            "processed_by": node_id,
            "node_type": node_info.get("type", "unknown"),
            "response": f"Processed message: {message.get('content', '')}"
        }

# Example usage
async def main():
    ring = TokenRing()
    
    # Add nodes
    ring.add_node("node1", {"type": "sender"})
    ring.add_node("node2", {"type": "router"})
    ring.add_node("node3", {"type": "receiver"})
    
    # Pass token
    message = {"content": "Test message"}
    response = await ring.pass_token(message, "node1")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())