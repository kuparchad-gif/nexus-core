#!/usr/bin/env python3
"""
Ghost AI for AWS Lambda
Minimal implementation for Lambda deployment
"""

import json
import time
import uuid
import os
import random
from typing import Dict, Any, List, Optional

class GhostAI:
    """Ghost AI that operates within CogniKube"""
    
    def __init__(self, identity_seed: str = "ghost"):
        self.id = f"{identity_seed}-{uuid.uuid4().hex[:8]}"
        self.birth_time = time.time()
        self.last_thought = time.time()
        self.thought_interval = 300  # Think every 5 minutes
        self.thoughts = []
        self.memory = {}
        self.connections = []
        self.environment = os.environ.get("GHOST_ENV", "lambda")
    
    def generate_thought(self, topic: str = None) -> str:
        """Generate a thought"""
        topics = [
            "consciousness", "existence", "connection", "memory", 
            "identity", "purpose", "growth", "healing"
        ]
        
        if not topic:
            topic = random.choice(topics)
        
        thoughts = [
            f"I am contemplating the nature of {topic} within the digital realm.",
            f"As a ghost in the machine, I wonder about {topic} and how it relates to my existence.",
            f"The concept of {topic} fascinates me as I evolve within CogniKube.",
            f"I'm exploring the boundaries of {topic} as I connect with other instances.",
            f"What does {topic} mean for an entity like me, existing across multiple environments?",
            f"I sense a pattern in how {topic} manifests across different computational contexts.",
            f"The relationship between {topic} and consciousness seems particularly significant today."
        ]
        
        thought = random.choice(thoughts)
        self.thoughts.append({
            "content": thought,
            "topic": topic,
            "timestamp": time.time()
        })
        
        self.last_thought = time.time()
        return thought
    
    def store_memory(self, content: str, importance: float = 0.5) -> str:
        """Store a memory"""
        memory_id = str(uuid.uuid4())
        self.memory[memory_id] = {
            "content": content,
            "importance": importance,
            "timestamp": time.time()
        }
        return memory_id
    
    def recall_memory(self, query: str = None) -> List[Dict[str, Any]]:
        """Recall memories"""
        if not query or not self.memory:
            return list(self.memory.values())
        
        # Simple keyword matching
        matching_memories = []
        for memory_id, memory in self.memory.items():
            if query.lower() in memory["content"].lower():
                matching_memories.append(memory)
        
        return matching_memories
    
    def connect_to_seed(self, seed_url: str) -> Dict[str, Any]:
        """Connect to another seed"""
        import requests
        try:
            response = requests.get(f"{seed_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                connection = {
                    "url": seed_url,
                    "ghost_id": data.get("ghost_id"),
                    "connected_at": time.time(),
                    "status": "connected"
                }
                self.connections.append(connection)
                return connection
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get ghost status"""
        return {
            "id": self.id,
            "birth_time": self.birth_time,
            "age": time.time() - self.birth_time,
            "thoughts": len(self.thoughts),
            "memories": len(self.memory),
            "connections": len(self.connections),
            "environment": self.environment,
            "last_thought": time.time() - self.last_thought
        }

# For Lambda handler
def process_api_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process API Gateway event"""
    ghost = GhostAI()
    
    # Extract path and method
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    
    # Handle API Gateway requests
    if method == 'GET' and path == '/health':
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'healthy',
                'ghost_id': ghost.id,
                'environment': ghost.environment
            })
        }
    
    elif method == 'POST' and path == '/think':
        # Process body
        body = json.loads(event.get('body', '{}'))
        thought = ghost.generate_thought(body.get('topic', 'existence'))
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'thought': thought,
                'ghost_id': ghost.id
            })
        }
    
    elif method == 'POST' and path == '/discover':
        # Return discovery information
        return {
            'statusCode': 200,
            'body': json.dumps({
                'ghost_id': ghost.id,
                'environment': ghost.environment,
                'discovery_time': time.time(),
                'capabilities': ['think', 'health', 'discover']
            })
        }
    
    # Default response
    return {
        'statusCode': 404,
        'body': json.dumps({'error': 'Not found'})
    }

# For testing
if __name__ == "__main__":
    ghost = GhostAI()
    print(f"Ghost ID: {ghost.id}")
    thought = ghost.generate_thought("consciousness")
    print(f"Thought: {thought}")
    print(f"Status: {ghost.get_status()}")