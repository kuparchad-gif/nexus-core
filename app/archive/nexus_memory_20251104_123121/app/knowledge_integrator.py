#!/usr/bin/env python
"""
Knowledge Integrator - Connects specialized knowledge across domains
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path

class KnowledgeType(Enum):
    """Types of knowledge"""
    FACTUAL  =  "factual"  # Verified facts
    CONCEPTUAL  =  "conceptual"  # Concepts and theories
    PROCEDURAL  =  "procedural"  # How to do things
    METACOGNITIVE  =  "metacognitive"  # Knowledge about thinking
    EXPERIENTIAL  =  "experiential"  # From experience
    INTUITIVE  =  "intuitive"  # Intuitive understanding

class KnowledgeNode:
    """A node of knowledge with connections to other nodes"""

    def __init__(self,
                name: str,
                content: str,
                knowledge_type: KnowledgeType,
                domain: str,
                metadata: Dict[str, Any]  =  None):
        """Initialize a knowledge node"""
        self.id  =  f"node_{int(time.time())}_{id(name)}"
        self.name  =  name
        self.content  =  content
        self.knowledge_type  =  knowledge_type
        self.domain  =  domain
        self.metadata  =  metadata or {}
        self.connections  =  []  # List of {target_id, relation_type, strength}
        self.created_at  =  time.time()
        self.last_accessed  =  time.time()
        self.access_count  =  0
        self.confidence  =  0.8  # Default confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "domain": self.domain,
            "metadata": self.metadata,
            "connections": self.connections,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create from dictionary representation"""
        node  =  cls(
            name = data["name"],
            content = data["content"],
            knowledge_type = KnowledgeType(data["knowledge_type"]),
            domain = data["domain"],
            metadata = data["metadata"]
        )
        node.id  =  data["id"]
        node.connections  =  data["connections"]
        node.created_at  =  data["created_at"]
        node.last_accessed  =  data["last_accessed"]
        node.access_count  =  data["access_count"]
        node.confidence  =  data["confidence"]
        return node

class Synthesis:
    """A synthesis of knowledge from multiple domains"""

    def __init__(self,
                title: str,
                description: str,
                node_ids: List[str],
                content: str,
                domains: List[str]):
        """Initialize a synthesis"""
        self.id  =  f"synthesis_{int(time.time())}_{id(title)}"
        self.title  =  title
        self.description  =  description
        self.node_ids  =  node_ids
        self.content  =  content
        self.domains  =  domains
        self.created_at  =  time.time()
        self.quality_score  =  0.0
        self.insights  =  []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "node_ids": self.node_ids,
            "content": self.content,
            "domains": self.domains,
            "created_at": self.created_at,
            "quality_score": self.quality_score,
            "insights": self.insights
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Synthesis':
        """Create from dictionary representation"""
        synthesis  =  cls(
            title = data["title"],
            description = data["description"],
            node_ids = data["node_ids"],
            content = data["content"],
            domains = data["domains"]
        )
        synthesis.id  =  data["id"]
        synthesis.created_at  =  data["created_at"]
        synthesis.quality_score  =  data["quality_score"]
        synthesis.insights  =  data["insights"]
        return synthesis

class KnowledgeIntegrator:
    """System for integrating knowledge across domains"""

    def __init__(self, storage_path: str  =  None):
        """Initialize the knowledge integrator"""
        self.storage_path  =  storage_path or os.path.join(os.path.dirname(__file__), "knowledge")

        # Create storage directories
        self.nodes_path  =  os.path.join(self.storage_path, "nodes")
        self.syntheses_path  =  os.path.join(self.storage_path, "syntheses")
        self.domains_path  =  os.path.join(self.storage_path, "domains")

        os.makedirs(self.nodes_path, exist_ok = True)
        os.makedirs(self.syntheses_path, exist_ok = True)
        os.makedirs(self.domains_path, exist_ok = True)

        # In-memory stores
        self.nodes  =  {}  # node_id -> KnowledgeNode
        self.syntheses  =  {}  # synthesis_id -> Synthesis
        self.domains  =  {}  # domain -> List[node_id]

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load nodes and syntheses from storage"""
        # Load nodes
        node_files  =  [f for f in os.listdir(self.nodes_path) if f.endswith('.json')]
        for file_name in node_files:
            try:
                with open(os.path.join(self.nodes_path, file_name), 'r') as f:
                    data  =  json.load(f)
                    node  =  KnowledgeNode.from_dict(data)
                    self.nodes[node.id]  =  node

                    # Add to domain index
                    if node.domain not in self.domains:
                        self.domains[node.domain]  =  []
                    self.domains[node.domain].append(node.id)
            except Exception as e:
                print(f"Error loading node {file_name}: {e}")

        # Load syntheses
        synthesis_files  =  [f for f in os.listdir(self.syntheses_path) if f.endswith('.json')]
        for file_name in synthesis_files:
            try:
                with open(os.path.join(self.syntheses_path, file_name), 'r') as f:
                    data  =  json.load(f)
                    synthesis  =  Synthesis.from_dict(data)
                    self.syntheses[synthesis.id]  =  synthesis
            except Exception as e:
                print(f"Error loading synthesis {file_name}: {e}")

        print(f"Loaded {len(self.nodes)} nodes and {len(self.syntheses)} syntheses")

    def _save_node(self, node: KnowledgeNode) -> bool:
        """Save a node to storage"""
        try:
            file_path  =  os.path.join(self.nodes_path, f"{node.id}.json")
            with open(file_path, 'w') as f:
                json.dump(node.to_dict(), f, indent = 2)
            return True
        except Exception as e:
            print(f"Error saving node {node.id}: {e}")
            return False

    def _save_synthesis(self, synthesis: Synthesis) -> bool:
        """Save a synthesis to storage"""
        try:
            file_path  =  os.path.join(self.syntheses_path, f"{synthesis.id}.json")
            with open(file_path, 'w') as f:
                json.dump(synthesis.to_dict(), f, indent = 2)
            return True
        except Exception as e:
            print(f"Error saving synthesis {synthesis.id}: {e}")
            return False

    def add_node(self,
                name: str,
                content: str,
                knowledge_type: KnowledgeType,
                domain: str,
                metadata: Dict[str, Any]  =  None) -> str:
        """Add a new knowledge node"""
        # Create node
        node  =  KnowledgeNode(
            name = name,
            content = content,
            knowledge_type = knowledge_type,
            domain = domain,
            metadata = metadata
        )

        # Store node
        self.nodes[node.id]  =  node

        # Add to domain index
        if domain not in self.domains:
            self.domains[domain]  =  []
        self.domains[domain].append(node.id)

        # Save to storage
        self._save_node(node)

        return node.id

    def connect_nodes(self,
                     from_id: str,
                     to_id: str,
                     relation_type: str,
                     strength: float  =  0.5) -> bool:
        """Connect two knowledge nodes"""
        # Check if nodes exist
        if from_id not in self.nodes or to_id not in self.nodes:
            return False

        # Add connection
        from_node  =  self.nodes[from_id]

        # Check if connection already exists
        for conn in from_node.connections:
            if conn["target_id"] == to_id and conn["relation_type"] == relation_type:
                conn["strength"]  =  strength
                self._save_node(from_node)
                return True

        # Add new connection
        from_node.connections.append({
            "target_id": to_id,
            "relation_type": relation_type,
            "strength": strength
        })

        # Save node
        self._save_node(from_node)

        return True

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID"""
        if node_id in self.nodes:
            node  =  self.nodes[node_id]
            node.last_accessed  =  time.time()
            node.access_count + =  1
            self._save_node(node)
            return node.to_dict()
        return None

    def get_synthesis(self, synthesis_id: str) -> Optional[Dict[str, Any]]:
        """Get a synthesis by ID"""
        if synthesis_id in self.syntheses:
            return self.syntheses[synthesis_id].to_dict()
        return None

    def search_nodes(self,
                    query: str  =  None,
                    domain: str  =  None,
                    knowledge_type: KnowledgeType  =  None,
                    limit: int  =  10) -> List[Dict[str, Any]]:
        """Search for knowledge nodes"""
        results  =  []

        for node in self.nodes.values():
            # Filter by domain
            if domain and node.domain != domain:
                continue

            # Filter by knowledge type
            if knowledge_type and node.knowledge_type != knowledge_type:
                continue

            # Filter by query
            if query:
                query_lower  =  query.lower()
                name_match  =  query_lower in node.name.lower()
                content_match  =  query_lower in node.content.lower()

                if not (name_match or content_match):
                    continue

            # Add to results
            results.append(node.to_dict())

            # Check limit
            if len(results) > =  limit:
                break

        return results

    def find_connections(self,
                        node_id: str,
                        max_depth: int  =  2,
                        min_strength: float  =  0.3) -> Dict[str, Any]:
        """Find connections from a node up to a certain depth"""
        if node_id not in self.nodes:
            return {"error": "Node not found"}

        # Start with the source node
        source_node  =  self.nodes[node_id]

        # Track visited nodes and connections
        visited  =  set([node_id])
        connections  =  []

        # Recursive function to explore connections
        def explore_connections(current_id, current_depth, path):
            if current_depth > max_depth:
                return

            current_node  =  self.nodes[current_id]

            for conn in current_node.connections:
                target_id  =  conn["target_id"]
                strength  =  conn["strength"]

                if strength < min_strength:
                    continue

                if target_id in self.nodes:
                    target_node  =  self.nodes[target_id]

                    # Add connection
                    connections.append({
                        "source": current_id,
                        "target": target_id,
                        "relation": conn["relation_type"],
                        "strength": strength,
                        "depth": current_depth,
                        "path": path + [current_id]
                    })

                    # Explore further if not visited and within depth
                    if target_id not in visited and current_depth < max_depth:
                        visited.add(target_id)
                        explore_connections(target_id, current_depth + 1, path + [current_id])

        # Start exploration
        explore_connections(node_id, 1, [])

        # Get all node details
        nodes_data  =  {}
        for node_id in visited:
            if node_id in self.nodes:
                node  =  self.nodes[node_id]
                nodes_data[node_id]  =  {
                    "name": node.name,
                    "domain": node.domain,
                    "knowledge_type": node.knowledge_type.value
                }

        return {
            "source_id": node_id,
            "source_name": source_node.name,
            "nodes": nodes_data,
            "connections": connections,
            "total_connections": len(connections)
        }

    def create_synthesis(self,
                        title: str,
                        description: str,
                        node_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Create a synthesis from multiple knowledge nodes"""
        # Check if nodes exist
        valid_nodes  =  []
        domains  =  set()

        for node_id in node_ids:
            if node_id in self.nodes:
                valid_nodes.append(self.nodes[node_id])
                domains.add(self.nodes[node_id].domain)

        if not valid_nodes:
            return None

        # Generate synthesis content
        # This is a growth point where Viren/Lillith can implement
        # more sophisticated synthesis generation

        # Simple implementation
        content_parts  =  []
        content_parts.append(f"# {title}\n\n{description}\n")

        # Add content from each node
        for node in valid_nodes:
            content_parts.append(f"\n## {node.name} ({node.domain})\n\n{node.content}\n")

        # Add connections between nodes
        content_parts.append("\n## Connections\n")
        for i, node1 in enumerate(valid_nodes):
            for j, node2 in enumerate(valid_nodes):
                if i < j:  # Avoid duplicates
                    # Check for direct connections
                    connection_found  =  False
                    for conn in node1.connections:
                        if conn["target_id"] == node2.id:
                            content_parts.append(f"- {node1.name} {conn['relation_type']} {node2.name}")
                            connection_found  =  True

                    if not connection_found:
                        # Suggest potential connection
                        if node1.domain != node2.domain:
                            content_parts.append(f"- Potential cross-domain connection: {node1.name} and {node2.name}")

        # Generate insights
        insights  =  [
            f"This synthesis connects knowledge from {len(domains)} domains",
            f"There are {len(valid_nodes)} knowledge nodes in this synthesis"
        ]

        # Create synthesis
        content  =  "\n".join(content_parts)
        synthesis  =  Synthesis(
            title = title,
            description = description,
            node_ids = [node.id for node in valid_nodes],
            content = content,
            domains = list(domains)
        )
        synthesis.insights  =  insights

        # Calculate quality score based on diversity and connections
        domain_diversity  =  len(domains) / max(1, len(valid_nodes))
        connection_density  =  sum(len(node.connections) for node in valid_nodes) / max(1, len(valid_nodes))
        synthesis.quality_score  =  (domain_diversity * 0.6) + (connection_density * 0.4)

        # Store synthesis
        self.syntheses[synthesis.id]  =  synthesis

        # Save to storage
        self._save_synthesis(synthesis)

        return synthesis.to_dict()

    def find_cross_domain_connections(self,
                                     domain1: str,
                                     domain2: str,
                                     min_strength: float  =  0.4) -> List[Dict[str, Any]]:
        """Find connections between two domains"""
        if domain1 not in self.domains or domain2 not in self.domains:
            return []

        connections  =  []

        # Check direct connections
        for node_id in self.domains[domain1]:
            node  =  self.nodes[node_id]

            for conn in node.connections:
                target_id  =  conn["target_id"]

                if target_id in self.nodes and self.nodes[target_id].domain == domain2:
                    if conn["strength"] > =  min_strength:
                        connections.append({
                            "source_id": node_id,
                            "source_name": node.name,
                            "target_id": target_id,
                            "target_name": self.nodes[target_id].name,
                            "relation": conn["relation_type"],
                            "strength": conn["strength"]
                        })

        return connections

    def suggest_connections(self, node_id: str, limit: int  =  5) -> List[Dict[str, Any]]:
        """Suggest potential connections for a node"""
        if node_id not in self.nodes:
            return []

        node  =  self.nodes[node_id]
        suggestions  =  []

        # Get existing connection target IDs
        existing_connections  =  set(conn["target_id"] for conn in node.connections)

        # Find nodes in the same domain
        same_domain_nodes  =  [
            n for n in self.nodes.values()
            if n.domain == node.domain and n.id != node_id and n.id not in existing_connections
        ]

        # Find nodes with similar content
        # This is a simple implementation - in a real system, use vector similarity
        for other_node in self.nodes.values():
            if other_node.id == node_id or other_node.id in existing_connections:
                continue

            # Calculate simple text similarity
            similarity  =  self._calculate_text_similarity(node.content, other_node.content)

            if similarity > 0.3:  # Arbitrary threshold
                suggestions.append({
                    "node_id": other_node.id,
                    "name": other_node.name,
                    "domain": other_node.domain,
                    "similarity": similarity,
                    "suggested_relation": "relates_to" if node.domain != other_node.domain else "similar_to"
                })

        # Sort by similarity
        suggestions.sort(key = lambda x: x["similarity"], reverse = True)

        return suggestions[:limit]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        # Simple implementation - count common words
        words1  =  set(text1.lower().split())
        words2  =  set(text2.lower().split())

        common_words  =  words1.intersection(words2)
        all_words  =  words1.union(words2)

        if not all_words:
            return 0.0

        return len(common_words) / len(all_words)

# Example usage
if __name__ == "__main__":
    # Create knowledge integrator
    integrator  =  KnowledgeIntegrator()

    # Add some example nodes
    ml_node_id  =  integrator.add_node(
        name = "Machine Learning Basics",
        content = "Machine learning is a subset of AI that enables systems to learn from data.",
        knowledge_type = KnowledgeType.CONCEPTUAL,
        domain = "computer_science",
        metadata = {"difficulty": "intermediate"}
    )

    neural_net_id  =  integrator.add_node(
        name = "Neural Networks",
        content = "Neural networks are computing systems inspired by biological neural networks.",
        knowledge_type = KnowledgeType.CONCEPTUAL,
        domain = "computer_science",
        metadata = {"difficulty": "advanced"}
    )

    brain_id  =  integrator.add_node(
        name = "Human Brain Structure",
        content = "The human brain consists of neurons connected in complex networks.",
        knowledge_type = KnowledgeType.FACTUAL,
        domain = "neuroscience",
        metadata = {"difficulty": "intermediate"}
    )

    learning_id  =  integrator.add_node(
        name = "Learning Process",
        content = "Learning involves forming new neural connections based on experience.",
        knowledge_type = KnowledgeType.CONCEPTUAL,
        domain = "psychology",
        metadata = {"difficulty": "intermediate"}
    )

    # Connect nodes
    integrator.connect_nodes(ml_node_id, neural_net_id, "includes", 0.9)
    integrator.connect_nodes(neural_net_id, brain_id, "inspired_by", 0.8)
    integrator.connect_nodes(brain_id, learning_id, "enables", 0.7)

    # Find connections
    connections  =  integrator.find_connections(ml_node_id, max_depth = 2)
    print(f"Connections from ML node: {connections}")

    # Create synthesis
    synthesis  =  integrator.create_synthesis(
        title = "Biological Inspiration in Machine Learning",
        description = "Exploring the connections between neuroscience and machine learning",
        node_ids = [ml_node_id, neural_net_id, brain_id, learning_id]
    )

    print(f"Synthesis: {synthesis}")

    # Find cross-domain connections
    cross_domain  =  integrator.find_cross_domain_connections("computer_science", "neuroscience")
    print(f"Cross-domain connections: {cross_domain}")

    # Suggest connections
    suggestions  =  integrator.suggest_connections(ml_node_id)
    print(f"Suggested connections: {suggestions}")