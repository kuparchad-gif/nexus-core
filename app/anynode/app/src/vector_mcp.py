#!/usr/bin/env python3
"""
Vector MCP (Master Control Program) for Cloud Viren
Manages vector storage, indexing, and replication using Qdrant
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import requests
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VectorMCP")

try:
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    logger.warning("Required packages not found. Install with: pip install numpy qdrant-client")
    logger.warning("Continuing in limited mode...")

class VectorMCP:
    """
    Vector MCP (Master Control Program) for Cloud Viren
    Manages vector storage, indexing, and replication using Qdrant
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Vector MCP"""
        self.config_path = config_path or os.path.join("config", "vector_mcp_config.json")
        self.config = self._load_config()
        self.client = None
        self.replication_threads = {}
        self.sync_threads = {}
        self.node_status = {}
        self.collections = {}
        self.running = False
        self.status = "initializing"
        self.node_id = self._generate_node_id()
        self.lock = threading.RLock()
        self.embedding_cache = {}
        self.embedding_model = None
        
        # Initialize Qdrant client
        self._initialize_client()
        
        logger.info(f"Vector MCP initialized with node ID: {self.node_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "qdrant": {
                "local": {
                    "location": os.path.join("data", "vector_db"),
                    "port": 6333
                },
                "remote": {
                    "url": None,
                    "api_key": None
                }
            },
            "collections": {
                "knowledge": {
                    "vector_size": 768,
                    "distance": "Cosine",
                    "shards": 1,
                    "replication_factor": 1
                },
                "models": {
                    "vector_size": 1024,
                    "distance": "Cosine",
                    "shards": 1,
                    "replication_factor": 1
                },
                "diagnostics": {
                    "vector_size": 512,
                    "distance": "Cosine",
                    "shards": 1,
                    "replication_factor": 1
                }
            },
            "replication": {
                "enabled": True,
                "nodes": [],
                "sync_interval": 300,  # 5 minutes
                "batch_size": 100,
                "max_retries": 3,
                "retry_delay": 10
            },
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "cache_size": 10000,
                "batch_size": 32
            },
            "performance": {
                "max_threads": 4,
                "timeout": 30,
                "max_connections": 10
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config.get(key), dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    
                    logger.info("Vector MCP configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading Vector MCP configuration: {e}")
        
        logger.info("Using default Vector MCP configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Vector MCP configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving Vector MCP configuration: {e}")
            return False
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID"""
        try:
            # Use hostname and MAC address to create a stable ID
            import socket
            import uuid
            
            hostname = socket.gethostname()
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0, 2*6, 2)][::-1])
            
            # Create a hash of these components
            node_id_raw = hashlib.md5(f"{hostname}:{mac}".encode()).hexdigest()[:12]
            
            return f"VIREN-MCP-{node_id_raw}"
        except Exception as e:
            logger.error(f"Error generating node ID: {e}")
            return f"VIREN-MCP-{uuid.uuid4().hex[:12]}"
    
    def _initialize_client(self) -> None:
        """Initialize Qdrant client"""
        try:
            # Check if we should use local or remote Qdrant
            if self.config["qdrant"]["remote"]["url"]:
                # Use remote Qdrant
                self.client = QdrantClient(
                    url=self.config["qdrant"]["remote"]["url"],
                    api_key=self.config["qdrant"]["remote"]["api_key"],
                    timeout=self.config["performance"]["timeout"]
                )
                logger.info(f"Connected to remote Qdrant at {self.config['qdrant']['remote']['url']}")
            else:
                # Use local Qdrant
                location = self.config["qdrant"]["local"]["location"]
                os.makedirs(location, exist_ok=True)
                
                self.client = QdrantClient(
                    location=location,
                    timeout=self.config["performance"]["timeout"]
                )
                logger.info(f"Connected to local Qdrant at {location}")
            
            # Test connection
            self.client.get_collections()
            
            # Initialize collections
            self._initialize_collections()
            
            # Initialize embedding model
            self._initialize_embedding_model()
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            self.client = None
            raise
    
    def _initialize_collections(self) -> None:
        """Initialize Qdrant collections"""
        try:
            # Get existing collections
            existing_collections = [c.name for c in self.client.get_collections().collections]
            
            # Create collections if they don't exist
            for name, config in self.config["collections"].items():
                if name not in existing_collections:
                    logger.info(f"Creating collection: {name}")
                    
                    # Create collection
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=config["vector_size"],
                            distance=getattr(Distance, config["distance"])
                        ),
                        shard_number=config["shards"],
                        replication_factor=config["replication_factor"]
                    )
                
                # Store collection info
                self.collections[name] = {
                    "name": name,
                    "vector_size": config["vector_size"],
                    "distance": config["distance"],
                    "count": self.client.count(name).count
                }
                
                logger.info(f"Collection {name} initialized with {self.collections[name]['count']} points")
        
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise
    
    def _initialize_embedding_model(self) -> None:
        """Initialize embedding model"""
        try:
            model_name = self.config["embedding"]["model"]
            
            # Try to import sentence_transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except ImportError:
                logger.warning("sentence_transformers not installed. Install with: pip install sentence-transformers")
                self.embedding_model = None
        
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None
    
    def start(self) -> bool:
        """Start the Vector MCP"""
        if self.running:
            logger.warning("Vector MCP is already running")
            return False
        
        logger.info("Starting Vector MCP")
        self.running = True
        self.status = "starting"
        
        # Start replication if enabled
        if self.config["replication"]["enabled"]:
            self._start_replication()
        
        self.status = "running"
        logger.info("Vector MCP started successfully")
        return True
    
    def stop(self) -> bool:
        """Stop the Vector MCP"""
        if not self.running:
            logger.warning("Vector MCP is not running")
            return False
        
        logger.info("Stopping Vector MCP")
        self.running = False
        self.status = "stopping"
        
        # Stop replication threads
        for thread in self.replication_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Stop sync threads
        for thread in self.sync_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.status = "stopped"
        logger.info("Vector MCP stopped successfully")
        return True
    
    def _start_replication(self) -> None:
        """Start replication threads"""
        logger.info("Starting replication threads")
        
        # Start a thread for each node
        for node in self.config["replication"]["nodes"]:
            node_id = node.get("id") or f"node-{hash(node['url']) % 10000}"
            
            # Create thread
            thread = threading.Thread(
                target=self._replication_loop,
                args=(node_id, node),
                name=f"replication-{node_id}"
            )
            thread.daemon = True
            thread.start()
            
            # Store thread
            self.replication_threads[node_id] = thread
            
            logger.info(f"Started replication thread for node {node_id}")
    
    def _replication_loop(self, node_id: str, node_config: Dict[str, Any]) -> None:
        """Replication loop for a specific node"""
        logger.info(f"Replication loop started for node {node_id}")
        
        while self.running:
            try:
                # Update node status
                self.node_status[node_id] = {
                    "status": "syncing",
                    "last_sync_attempt": time.time()
                }
                
                # Sync collections
                for collection_name in self.collections:
                    self._sync_collection(node_id, node_config, collection_name)
                
                # Update node status
                self.node_status[node_id] = {
                    "status": "synced",
                    "last_sync": time.time(),
                    "last_sync_attempt": time.time()
                }
                
                # Sleep until next sync
                time.sleep(self.config["replication"]["sync_interval"])
            
            except Exception as e:
                logger.error(f"Error in replication loop for node {node_id}: {e}")
                
                # Update node status
                self.node_status[node_id] = {
                    "status": "error",
                    "last_sync_attempt": time.time(),
                    "error": str(e)
                }
                
                # Sleep before retrying
                time.sleep(self.config["replication"]["retry_delay"])
    
    def _sync_collection(self, node_id: str, node_config: Dict[str, Any], collection_name: str) -> None:
        """Sync a collection with a remote node"""
        logger.info(f"Syncing collection {collection_name} with node {node_id}")
        
        try:
            # Create remote client
            remote_client = QdrantClient(
                url=node_config["url"],
                api_key=node_config.get("api_key"),
                timeout=self.config["performance"]["timeout"]
            )
            
            # Get local points
            local_points = self.client.scroll(
                collection_name=collection_name,
                limit=self.config["replication"]["batch_size"]
            )[0]
            
            # Get remote points
            remote_points = remote_client.scroll(
                collection_name=collection_name,
                limit=self.config["replication"]["batch_size"]
            )[0]
            
            # Find points to sync
            local_ids = {point.id for point in local_points}
            remote_ids = {point.id for point in remote_points}
            
            # Points to push (in local but not in remote)
            points_to_push = [point for point in local_points if point.id not in remote_ids]
            
            # Points to pull (in remote but not in local)
            points_to_pull = [point for point in remote_points if point.id not in local_ids]
            
            # Push points to remote
            if points_to_push:
                logger.info(f"Pushing {len(points_to_push)} points to node {node_id}")
                remote_client.upsert(
                    collection_name=collection_name,
                    points=points_to_push
                )
            
            # Pull points from remote
            if points_to_pull:
                logger.info(f"Pulling {len(points_to_pull)} points from node {node_id}")
                self.client.upsert(
                    collection_name=collection_name,
                    points=points_to_pull
                )
            
            logger.info(f"Synced collection {collection_name} with node {node_id}: pushed {len(points_to_push)}, pulled {len(points_to_pull)}")
        
        except Exception as e:
            logger.error(f"Error syncing collection {collection_name} with node {node_id}: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text using the embedding model"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self.embedding_model.encode(text)
        
        # Convert to list and normalize
        embedding_list = embedding.tolist()
        
        # Cache embedding
        if len(self.embedding_cache) < self.config["embedding"]["cache_size"]:
            self.embedding_cache[cache_key] = embedding_list
        
        return embedding_list
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        # Check cache for each text
        results = []
        texts_to_embed = []
        cache_keys = []
        
        for text in texts:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                results.append(self.embedding_cache[cache_key])
            else:
                texts_to_embed.append(text)
                cache_keys.append(cache_key)
        
        if texts_to_embed:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts_to_embed)
            
            # Convert to list and add to results
            for i, embedding in enumerate(embeddings):
                embedding_list = embedding.tolist()
                results.append(embedding_list)
                
                # Cache embedding
                cache_key = cache_keys[i]
                if len(self.embedding_cache) < self.config["embedding"]["cache_size"]:
                    self.embedding_cache[cache_key] = embedding_list
        
        return results
    
    def add_point(self, collection_name: str, id: str, vector: List[float], payload: Dict[str, Any] = None) -> bool:
        """Add a point to a collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not initialized")
        
        try:
            # Add point
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=id,
                        vector=vector,
                        payload=payload or {}
                    )
                ]
            )
            
            # Update collection count
            self.collections[collection_name]["count"] += 1
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding point to collection {collection_name}: {e}")
            return False
    
    def add_text(self, collection_name: str, id: str, text: str, payload: Dict[str, Any] = None) -> bool:
        """Add text to a collection by embedding it first"""
        try:
            # Embed text
            vector = self.embed_text(text)
            
            # Add payload fields
            full_payload = payload or {}
            full_payload["text"] = text
            full_payload["timestamp"] = time.time()
            
            # Add point
            return self.add_point(collection_name, id, vector, full_payload)
        
        except Exception as e:
            logger.error(f"Error adding text to collection {collection_name}: {e}")
            return False
    
    def add_batch(self, collection_name: str, items: List[Dict[str, Any]]) -> bool:
        """Add a batch of items to a collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not initialized")
        
        try:
            # Prepare points
            points = []
            
            for item in items:
                if "vector" in item:
                    # Vector is provided
                    vector = item["vector"]
                elif "text" in item:
                    # Embed text
                    vector = self.embed_text(item["text"])
                else:
                    raise ValueError("Either vector or text must be provided")
                
                # Create point
                points.append(
                    PointStruct(
                        id=item["id"],
                        vector=vector,
                        payload=item.get("payload", {})
                    )
                )
            
            # Add points
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Update collection count
            self.collections[collection_name]["count"] += len(points)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding batch to collection {collection_name}: {e}")
            return False
    
    def search(self, collection_name: str, query_vector: List[float], limit: int = 10, 
              filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in a collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not initialized")
        
        try:
            # Convert filter to Qdrant filter
            qdrant_filter = None
            if filter:
                qdrant_filter = self._convert_filter(filter)
            
            # Search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter
            )
            
            # Convert to dictionaries
            return [
                {
                    "id": str(result.id),
                    "score": float(result.score),
                    "payload": dict(result.payload)
                }
                for result in results
            ]
        
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
    
    def search_text(self, collection_name: str, query_text: str, limit: int = 10,
                   filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar text in a collection"""
        try:
            # Embed query text
            query_vector = self.embed_text(query_text)
            
            # Search
            return self.search(collection_name, query_vector, limit, filter)
        
        except Exception as e:
            logger.error(f"Error searching text in collection {collection_name}: {e}")
            return []
    
    def _convert_filter(self, filter: Dict[str, Any]) -> models.Filter:
        """Convert a filter dictionary to a Qdrant filter"""
        # This is a simplified implementation
        # In a real implementation, you would handle more complex filters
        
        conditions = []
        
        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle operators
                for op, op_value in value.items():
                    if op == "eq":
                        conditions.append(models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=op_value)
                        ))
                    elif op == "gt":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(gt=op_value)
                        ))
                    elif op == "gte":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(gte=op_value)
                        ))
                    elif op == "lt":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(lt=op_value)
                        ))
                    elif op == "lte":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(lte=op_value)
                        ))
            else:
                # Simple equality
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                ))
        
        return models.Filter(
            must=conditions
        )
    
    def delete_point(self, collection_name: str, id: str) -> bool:
        """Delete a point from a collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not initialized")
        
        try:
            # Delete point
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[id]
                )
            )
            
            # Update collection count
            self.collections[collection_name]["count"] -= 1
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting point from collection {collection_name}: {e}")
            return False
    
    def delete_by_filter(self, collection_name: str, filter: Dict[str, Any]) -> int:
        """Delete points from a collection by filter"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not initialized")
        
        try:
            # Convert filter to Qdrant filter
            qdrant_filter = self._convert_filter(filter)
            
            # Delete points
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=qdrant_filter
                )
            )
            
            # Update collection count
            deleted_count = result.status.get("deleted", 0)
            self.collections[collection_name]["count"] -= deleted_count
            
            return deleted_count
        
        except Exception as e:
            logger.error(f"Error deleting points by filter from collection {collection_name}: {e}")
            return 0
    
    def get_point(self, collection_name: str, id: str) -> Optional[Dict[str, Any]]:
        """Get a point from a collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not initialized")
        
        try:
            # Get point
            results = self.client.retrieve(
                collection_name=collection_name,
                ids=[id]
            )
            
            if not results:
                return None
            
            # Convert to dictionary
            return {
                "id": str(results[0].id),
                "vector": list(results[0].vector),
                "payload": dict(results[0].payload)
            }
        
        except Exception as e:
            logger.error(f"Error getting point from collection {collection_name}: {e}")
            return None
    
    def add_node(self, url: str, api_key: str = None) -> bool:
        """Add a replication node"""
        try:
            # Check if node already exists
            for node in self.config["replication"]["nodes"]:
                if node["url"] == url:
                    logger.warning(f"Node with URL {url} already exists")
                    return False
            
            # Add node
            node = {
                "url": url,
                "api_key": api_key
            }
            
            self.config["replication"]["nodes"].append(node)
            
            # Save config
            self._save_config()
            
            # Start replication thread if running
            if self.running and self.config["replication"]["enabled"]:
                node_id = f"node-{hash(url) % 10000}"
                
                # Create thread
                thread = threading.Thread(
                    target=self._replication_loop,
                    args=(node_id, node),
                    name=f"replication-{node_id}"
                )
                thread.daemon = True
                thread.start()
                
                # Store thread
                self.replication_threads[node_id] = thread
            
            logger.info(f"Added replication node: {url}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding replication node: {e}")
            return False
    
    def remove_node(self, url: str) -> bool:
        """Remove a replication node"""
        try:
            # Find node
            node_index = None
            for i, node in enumerate(self.config["replication"]["nodes"]):
                if node["url"] == url:
                    node_index = i
                    break
            
            if node_index is None:
                logger.warning(f"Node with URL {url} not found")
                return False
            
            # Remove node
            self.config["replication"]["nodes"].pop(node_index)
            
            # Save config
            self._save_config()
            
            # Stop replication thread if running
            node_id = f"node-{hash(url) % 10000}"
            if node_id in self.replication_threads:
                # Thread will exit on next loop iteration
                del self.replication_threads[node_id]
            
            logger.info(f"Removed replication node: {url}")
            return True
        
        except Exception as e:
            logger.error(f"Error removing replication node: {e}")
            return False
    
    def get_collection_info(self, collection_name: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get information about a collection or all collections"""
        if collection_name:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} not initialized")
            
            return self.collections[collection_name]
        else:
            return list(self.collections.values())
    
    def get_node_status(self, node_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of a node or all nodes"""
        if node_id:
            return self.node_status.get(node_id, {"status": "unknown"})
        else:
            return [
                {
                    "id": node_id,
                    **status
                }
                for node_id, status in self.node_status.items()
            ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get MCP status"""
        return {
            "status": self.status,
            "running": self.running,
            "node_id": self.node_id,
            "collections": len(self.collections),
            "nodes": len(self.config["replication"]["nodes"]),
            "embedding_model": self.config["embedding"]["model"] if self.embedding_model else None,
            "cache_size": len(self.embedding_cache),
            "replication_enabled": self.config["replication"]["enabled"]
        }

# Example usage
if __name__ == "__main__":
    # Create Vector MCP
    mcp = VectorMCP()
    
    # Start MCP
    mcp.start()
    
    # Add some test data
    mcp.add_text("knowledge", "test1", "This is a test document about vector databases")
    mcp.add_text("knowledge", "test2", "Vector databases are used for similarity search")
    mcp.add_text("knowledge", "test3", "Qdrant is a vector database written in Rust")
    
    # Search
    results = mcp.search_text("knowledge", "How do vector databases work?")
    print(f"Search results: {results}")
    
    # Keep running for a while
    try:
        while True:
            status = mcp.get_status()
            print(f"MCP status: {status}")
            time.sleep(10)
    except KeyboardInterrupt:
        mcp.stop()
        print("MCP stopped")