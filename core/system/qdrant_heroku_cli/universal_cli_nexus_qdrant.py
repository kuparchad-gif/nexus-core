#!/usr/bin/env python3
# universal_cli_nexus_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import *
from sentence_transformers import SentenceTransformer
import json
from typing import Dict, List, Optional
import hashlib
from datetime import datetime

class QdrantCLINexus:
    """Universal CLI Knowledge Base with Qdrant Vector Search"""
    
    def __init__(self, qdrant_url: str = "localhost:6333"):
        self.client = QdrantClient(qdrant_url)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._setup_collections()
        
    def _setup_collections(self):
        """Setup Qdrant collections for CLI knowledge"""
        collections = {
            "cli_commands": 384,  # Command embeddings
            "language_patterns": 256,  # Language syntax patterns
            "porting_rules": 512,  # Cross-language translation rules
            "usage_context": 384   # Usage context embeddings
        }
        
        for coll, size in collections.items():
            try:
                self.client.get_collection(coll)
            except:
                self.client.create_collection(
                    coll,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE)
                )
    
    def store_command(self, command_data: Dict):
        """Store a CLI command with semantic embedding"""
        # Create semantic embedding from command + description
        semantic_text = f"{command_data['command']} {command_data.get('description', '')} {command_data.get('language', '')}"
        embedding = self.embedder.encode(semantic_text).tolist()
        
        point_id = self._generate_command_id(command_data)
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "type": "cli_command",
                "command": command_data["command"],
                "language": command_data["language"],
                "version": command_data.get("version", "latest"),
                "syntax": command_data["syntax"],
                "examples": command_data.get("examples", []),
                "description": command_data.get("description", ""),
                "tags": command_data.get("tags", []),
                "similar_commands": command_data.get("similar_commands", []),
                "timestamp": datetime.now().isoformat(),
                "usage_count": 0
            }
        )
        
        self.client.upsert("cli_commands", [point])
        return point_id
    
    def find_similar_commands(self, query: str, language: str = None, limit: int = 5) -> List[Dict]:
        """Find commands by semantic similarity"""
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build filter
        filter_condition = None
        if language:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=language)
                    )
                ]
            )
        
        search_result = self.client.search(
            collection_name="cli_commands",
            query_vector=query_embedding,
            query_filter=filter_condition,
            limit=limit
        )
        
        return [
            {
                "command": hit.payload["command"],
                "language": hit.payload["language"],
                "syntax": hit.payload["syntax"],
                "similarity_score": hit.score,
                "examples": hit.payload.get("examples", [])
            }
            for hit in search_result
        ]
    
    def port_command_across_languages(self, source_command: str, target_language: str) -> List[Dict]:
        """Port commands across languages using semantic similarity"""
        # Find the source command
        source_results = self.find_similar_commands(source_command, limit=1)
        if not source_results:
            return []
        
        source_cmd = source_results[0]
        
        # Find similar patterns in target language
        semantic_query = f"{source_cmd['command']} {source_cmd.get('description', '')}"
        target_results = self.find_similar_commands(semantic_query, target_language, limit=3)
        
        ported_commands = []
        for result in target_results:
            confidence = self._calculate_porting_confidence(source_cmd, result)
            ported_commands.append({
                **result,
                "ported_from": source_cmd["language"],
                "original_command": source_cmd["command"],
                "porting_confidence": confidence
            })
        
        return ported_commands
    
    def _calculate_porting_confidence(self, source: Dict, target: Dict) -> float:
        """Calculate confidence score for command porting"""
        # Compare semantic similarity of descriptions
        source_desc = source.get('description', '')
        target_desc = target.get('description', '')
        
        if source_desc and target_desc:
            desc_similarity = self.embedder.similarity(
                self.embedder.encode([source_desc]),
                self.embedder.encode([target_desc])
            )[0][0]
            return float(desc_similarity)
        
        return 0.5  # Default medium confidence

class DynamicCLICompleter:
    """Dynamic CLI completion using Qdrant nexus"""
    
    def __init__(self, nexus: QdrantCLINexus):
        self.nexus = nexus
        self.session_history = []
        
    def suggest_completions(self, partial_input: str, context: Dict = None) -> List[Dict]:
        """Suggest command completions based on partial input and context"""
        # Get semantic suggestions
        semantic_suggestions = self.nexus.find_similar_commands(partial_input, limit=10)
        
        # Filter by context if provided
        if context and context.get('language'):
            semantic_suggestions = [
                cmd for cmd in semantic_suggestions 
                if cmd['language'] == context['language']
            ]
        
        # Add usage-based ranking
        ranked_suggestions = self._rank_suggestions(semantic_suggestions, partial_input)
        
        return ranked_suggestions[:5]  # Return top 5
    
    def _rank_suggestions(self, suggestions: List[Dict], partial_input: str) -> List[Dict]:
        """Rank suggestions by relevance"""
        ranked = []
        for suggestion in suggestions:
            score = suggestion['similarity_score']
            
            # Boost exact prefix matches
            if suggestion['command'].startswith(partial_input):
                score *= 1.5
                
            # Boost commands from current session history
            if suggestion['command'] in self.session_history:
                score *= 1.2
                
            ranked.append({**suggestion, 'final_score': score})
        
        return sorted(ranked, key=lambda x: x['final_score'], reverse=True)

# Example usage with real CLI commands
def populate_sample_data(nexus: QdrantCLINexus):
    """Populate with sample CLI commands across languages"""
    
    sample_commands = [
        {
            "command": "git commit",
            "language": "git",
            "syntax": "git commit -m \"message\"",
            "description": "Commit staged changes to repository",
            "examples": [
                "git commit -m \"Initial commit\"",
                "git commit -am \"Quick commit all changes\""
            ],
            "tags": ["version-control", "commit", "repository"]
        },
        {
            "command": "docker build",
            "language": "docker", 
            "syntax": "docker build -t tag path",
            "description": "Build Docker image from Dockerfile",
            "examples": [
                "docker build -t myapp .",
                "docker build -f Dockerfile.prod -t production ."
            ],
            "tags": ["container", "build", "image"]
        },
        {
            "command": "kubectl apply",
            "language": "kubectl",
            "syntax": "kubectl apply -f filename",
            "description": "Apply configuration to resource",
            "examples": [
                "kubectl apply -f deployment.yaml",
                "kubectl apply -k path/to/kustomize"
            ],
            "tags": ["kubernetes", "deployment", "apply"]
        }
    ]
    
    for cmd in sample_commands:
        nexus.store_command(cmd)

if __name__ == "__main__":
    # Initialize Qdrant CLI Nexus
    nexus = QdrantCLINexus()
    
    # Populate with sample data
    populate_sample_data(nexus)
    
    # Test semantic search
    print("=== Semantic Command Search ===")
    results = nexus.find_similar_commands("build container image", language="docker")
    for result in results:
        print(f"{result['command']} (score: {result['similarity_score']:.3f})")
        print(f"  Syntax: {result['syntax']}")
    
    print("\n=== Cross-Language Porting ===")
    ported = nexus.port_command_across_languages("docker build", "kubernetes")
    for p in ported:
        print(f"{p['command']} (confidence: {p['porting_confidence']:.3f})")
    
    # Test dynamic completion
    print("\n=== Dynamic Completion ===")
    completer = DynamicCLICompleter(nexus)
    suggestions = completer.suggest_completions("git com")
    for sug in suggestions:
        print(f"{sug['command']} (score: {sug['final_score']:.3f})")