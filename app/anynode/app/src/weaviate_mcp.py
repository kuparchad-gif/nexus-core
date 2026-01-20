#!/usr/bin/env python3
"""
Weaviate MCP for Cloud Viren
Manages vector storage, indexing, and replication using Weaviate
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WeaviateMCP")

try:
    import weaviate
    from weaviate.embedded import EmbeddedOptions
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("Required packages not found. Install with: pip install weaviate-client sentence-transformers numpy")
    logger.warning("Continuing in limited mode...")

class WeaviateMCP:
    """
    Weaviate MCP for Cloud Viren
    Manages vector storage, indexing, and replication using Weaviate
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Weaviate MCP"""
        self.config_path = config_path or os.path.join("config", "weaviate_mcp_config.json")
        self.config = self._load_config()
        self.client = None
        self.replication_threads = {}
        self.node_status = {}
        self.collections = {}
        self.running = False
        self.status = "initializing"
        self.node_id = self._generate_node_id()
        self.lock = threading.RLock()
        self.embedding_cache = {}
        self.embedding_model = None
        
        # Initialize Weaviate client
        self._initialize_client()
        
        logger.info(f"Weaviate MCP initialized with node ID: {self.node_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "weaviate": {
                "local": {
                    "enabled": True,
                    "port": 8080,
                    "persistence_path": os.path.join("data", "weaviate_data")
                },
                "remote": {
                    "url": None,
                    "api_key": None
                }
            },
            "classes": {
                "Knowledge": {
                    "description": "Knowledge base for Cloud Viren",
                    "vectorizer": "text2vec-transformers",
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "source", "dataType": ["string"]},
                        {"name": "topic", "dataType": ["string"]},
                        {"name": "timestamp", "dataType": ["number"]}
                    ]
                },
                "Models": {
                    "description": "Model information for Cloud Viren",
                    "vectorizer": "text2vec-transformers",
                    "properties": [
                        {"name": "description", "dataType": ["text"]},
                        {"name": "model_type", "dataType": ["string"]},
                        {"name": "model_size", "dataType": ["string"]},
                        {"name": "parameters", "dataType": ["number"]},
                        {"name": "timestamp", "dataType": ["number"]}
                    ]
                },
                "Diagnostics": {
                    "description": "Diagnostic information for Cloud Viren",
                    "vectorizer": "text2vec-transformers",
                    "properties": [
                        {"name": "description", "dataType": ["text"]},
                        {"name": "component", "dataType": ["string"]},
                        {"name": "status", "dataType": ["string"]},
                        {"name": "severity", "dataType": ["string"]},
                        {"name": "timestamp", "dataType": ["number"]}
                    ]
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
            },
            "boot_intelligence": {
                "enabled": True,
                "bootstrap_knowledge": True,
                "self_learning": True,
                "memory_consolidation": True,
                "consolidation_interval": 86400  # 24 hours
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
                    
                    logger.info("Weaviate MCP configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading Weaviate MCP configuration: {e}")
        
        logger.info("Using default Weaviate MCP configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Weaviate MCP configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving Weaviate MCP configuration: {e}")
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
            import hashlib
            node_id_raw = hashlib.md5(f"{hostname}:{mac}".encode()).hexdigest()[:12]
            
            return f"VIREN-MCP-{node_id_raw}"
        except Exception as e:
            logger.error(f"Error generating node ID: {e}")
            return f"VIREN-MCP-{uuid.uuid4().hex[:12]}"
    
    def _initialize_client(self) -> None:
        """Initialize Weaviate client"""
        try:
            # Check if we should use local or remote Weaviate
            if self.config["weaviate"]["remote"]["url"]:
                # Use remote Weaviate
                auth_config = weaviate.auth.AuthApiKey(api_key=self.config["weaviate"]["remote"]["url"])
                
                self.client = weaviate.Client(
                    url=self.config["weaviate"]["remote"]["url"],
                    auth_client_secret=auth_config if self.config["weaviate"]["remote"]["api_key"] else None,
                    timeout_config=weaviate.client.TimeoutConfig(
                        timeout_config=self.config["performance"]["timeout"]
                    )
                )
                logger.info(f"Connected to remote Weaviate at {self.config['weaviate']['remote']['url']}")
            else:
                # Use local embedded Weaviate
                persistence_path = self.config["weaviate"]["local"]["persistence_path"]
                os.makedirs(persistence_path, exist_ok=True)
                
                self.client = weaviate.Client(
                    embedded_options=EmbeddedOptions(
                        persistence_data_path=persistence_path,
                        port=self.config["weaviate"]["local"]["port"]
                    )
                )
                logger.info(f"Started embedded Weaviate at port {self.config['weaviate']['local']['port']}")
            
            # Initialize schema
            self._initialize_schema()
            
            # Initialize embedding model
            self._initialize_embedding_model()
            
            # Initialize boot intelligence
            if self.config["boot_intelligence"]["enabled"]:
                self._initialize_boot_intelligence()
            
        except Exception as e:
            logger.error(f"Error initializing Weaviate client: {e}")
            self.client = None
            raise
    
    def _initialize_schema(self) -> None:
        """Initialize Weaviate schema"""
        try:
            # Get existing classes
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]
            
            # Create classes if they don't exist
            for class_name, class_config in self.config["classes"].items():
                if class_name not in existing_classes:
                    logger.info(f"Creating class: {class_name}")
                    
                    # Create class object
                    class_obj = {
                        "class": class_name,
                        "description": class_config["description"],
                        "vectorizer": class_config["vectorizer"],
                        "properties": class_config["properties"]
                    }
                    
                    # Create class
                    self.client.schema.create_class(class_obj)
                
                # Store class info
                self.collections[class_name] = {
                    "name": class_name,
                    "description": class_config["description"],
                    "count": self._get_collection_count(class_name)
                }
                
                logger.info(f"Class {class_name} initialized with {self.collections[class_name]['count']} objects")
        
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise
    
    def _get_collection_count(self, class_name: str) -> int:
        """Get count of objects in a collection"""
        try:
            result = self.client.query.aggregate(class_name).with_meta_count().do()
            return result["data"]["Aggregate"][class_name][0]["meta"]["count"]
        except Exception as e:
            logger.error(f"Error getting count for class {class_name}: {e}")
            return 0
    
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
    
    def _initialize_boot_intelligence(self) -> None:
        """Initialize boot intelligence like Lillith"""
        logger.info("Initializing boot intelligence")
        
        try:
            # Bootstrap knowledge if enabled
            if self.config["boot_intelligence"]["bootstrap_knowledge"]:
                self._bootstrap_knowledge()
            
            # Start memory consolidation thread if enabled
            if self.config["boot_intelligence"]["memory_consolidation"]:
                consolidation_thread = threading.Thread(target=self._memory_consolidation_loop)
                consolidation_thread.daemon = True
                consolidation_thread.start()
                logger.info("Memory consolidation thread started")
        
        except Exception as e:
            logger.error(f"Error initializing boot intelligence: {e}")
    
    def _bootstrap_knowledge(self) -> None:
        """Bootstrap knowledge base with essential information"""
        logger.info("Bootstrapping knowledge base")
        
        # Check if knowledge base is empty
        if self._get_collection_count("Knowledge") > 0:
            logger.info("Knowledge base already contains data, skipping bootstrap")
            return
        
        # Add essential knowledge
        bootstrap_knowledge = [
            {
                "content": "Cloud Viren is an advanced AI system with a model cascade from 1B to 256B parameters.",
                "source": "system",
                "topic": "overview"
            },
            {
                "content": "The Weaviate MCP uses a vector database for knowledge storage and retrieval.",
                "source": "system",
                "topic": "architecture"
            },
            {
                "content": "RAG (Retrieval-Augmented Generation) enhances model responses with retrieved information.",
                "source": "system",
                "topic": "rag"
            },
            {
                "content": "The model cascade includes models of sizes: 1B, 3B, 7B, 14B, 27B, 128B, and 256B.",
                "source": "system",
                "topic": "models"
            },
            {
                "content": "Weaviate provides GraphQL-based querying and multimodal data storage capabilities.",
                "source": "system",
                "topic": "database"
            }
        ]
        
        # Add bootstrap knowledge
        for knowledge in bootstrap_knowledge:
            self.add_knowledge(
                content=knowledge["content"],
                metadata={
                    "source": knowledge["source"],
                    "topic": knowledge["topic"]
                }
            )
        
        logger.info(f"Added {len(bootstrap_knowledge)} bootstrap knowledge entries")
    
    def _memory_consolidation_loop(self) -> None:
        """Memory consolidation loop for boot intelligence"""
        logger.info("Starting memory consolidation loop")
        
        while self.running:
            try:
                # Sleep until next consolidation
                time.sleep(self.config["boot_intelligence"]["consolidation_interval"])
                
                if not self.running:
                    break
                
                logger.info("Running memory consolidation")
                
                # Perform consolidation
                self._consolidate_memory()
                
            except Exception as e:
                logger.error(f"Error in memory consolidation loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _consolidate_memory(self) -> None:
        """Consolidate memory by analyzing and synthesizing knowledge"""
        try:
            # Get all knowledge entries
            result = self.client.query.get(
                "Knowledge", ["content", "source", "topic", "timestamp"]
            ).with_limit(1000).do()
            
            if "data" not in result or "Get" not in result["data"] or "Knowledge" not in result["data"]["Get"]:
                logger.warning("No knowledge entries found for consolidation")
                return
            
            knowledge_entries = result["data"]["Get"]["Knowledge"]
            
            # Group by topic
            topics = {}
            for entry in knowledge_entries:
                topic = entry.get("topic", "general")
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(entry)
            
            # For each topic with multiple entries, create a consolidated entry
            for topic, entries in topics.items():
                if len(entries) < 3:
                    continue  # Need at least 3 entries to consolidate
                
                # Sort by timestamp if available
                entries.sort(key=lambda x: x.get("timestamp", 0))
                
                # Combine content
                combined_content = "\n\n".join([e["content"] for e in entries])
                
                # If we have an embedding model, we could summarize here
                # For now, just use the combined content
                consolidated_content = f"Consolidated knowledge about {topic}:\n\n{combined_content}"
                
                # Add consolidated knowledge
                self.add_knowledge(
                    content=consolidated_content,
                    metadata={
                        "source": "consolidation",
                        "topic": topic,
                        "consolidated_from": len(entries)
                    }
                )
                
                logger.info(f"Consolidated {len(entries)} entries for topic: {topic}")
        
        except Exception as e:
            logger.error(f"Error consolidating memory: {e}")
    
    def start(self) -> bool:
        """Start the Weaviate MCP"""
        if self.running:
            logger.warning("Weaviate MCP is already running")
            return False
        
        logger.info("Starting Weaviate MCP")
        self.running = True
        self.status = "starting"
        
        # Start replication if enabled
        if self.config["replication"]["enabled"]:
            self._start_replication()
        
        self.status = "running"
        logger.info("Weaviate MCP started successfully")
        return True
    
    def stop(self) -> bool:
        """Stop the Weaviate MCP"""
        if not self.running:
            logger.warning("Weaviate MCP is not running")
            return False
        
        logger.info("Stopping Weaviate MCP")
        self.running = False
        self.status = "stopping"
        
        # Stop replication threads
        for thread in self.replication_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.status = "stopped"
        logger.info("Weaviate MCP stopped successfully")
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
                for class_name in self.collections:
                    self._sync_collection(node_id, node_config, class_name)
                
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
    
    def _sync_collection(self, node_id: str, node_config: Dict[str, Any], class_name: str) -> None:
        """Sync a collection with a remote node"""
        logger.info(f"Syncing collection {class_name} with node {node_id}")
        
        try:
            # Create remote client
            auth_config = weaviate.auth.AuthApiKey(api_key=node_config.get("api_key")) if node_config.get("api_key") else None
            
            remote_client = weaviate.Client(
                url=node_config["url"],
                auth_client_secret=auth_config,
                timeout_config=weaviate.client.TimeoutConfig(
                    timeout_config=self.config["performance"]["timeout"]
                )
            )
            
            # Get last sync timestamp
            last_sync = self.node_status.get(node_id, {}).get("last_sync", 0)
            
            # Get local objects updated since last sync
            local_objects = self._get_objects_since(class_name, last_sync)
            
            # Get remote objects updated since last sync
            remote_objects = self._get_remote_objects_since(remote_client, class_name, last_sync)
            
            # Find objects to sync
            local_ids = {obj["id"] for obj in local_objects}
            remote_ids = {obj["id"] for obj in remote_objects}
            
            # Objects to push (in local but not in remote or updated in local)
            objects_to_push = [obj for obj in local_objects if obj["id"] not in remote_ids]
            
            # Objects to pull (in remote but not in local or updated in remote)
            objects_to_pull = [obj for obj in remote_objects if obj["id"] not in local_ids]
            
            # Push objects to remote
            if objects_to_push:
                logger.info(f"Pushing {len(objects_to_push)} objects to node {node_id}")
                for obj in objects_to_push:
                    properties = {k: v for k, v in obj.items() if k not in ["id", "vector"]}
                    
                    # Add object to remote
                    if "vector" in obj:
                        remote_client.data_object.create(
                            data_object=properties,
                            class_name=class_name,
                            uuid=obj["id"],
                            vector=obj["vector"]
                        )
                    else:
                        remote_client.data_object.create(
                            data_object=properties,
                            class_name=class_name,
                            uuid=obj["id"]
                        )
            
            # Pull objects from remote
            if objects_to_pull:
                logger.info(f"Pulling {len(objects_to_pull)} objects from node {node_id}")
                for obj in objects_to_pull:
                    properties = {k: v for k, v in obj.items() if k not in ["id", "vector"]}
                    
                    # Add object to local
                    if "vector" in obj:
                        self.client.data_object.create(
                            data_object=properties,
                            class_name=class_name,
                            uuid=obj["id"],
                            vector=obj["vector"]
                        )
                    else:
                        self.client.data_object.create(
                            data_object=properties,
                            class_name=class_name,
                            uuid=obj["id"]
                        )
            
            logger.info(f"Synced collection {class_name} with node {node_id}: pushed {len(objects_to_push)}, pulled {len(objects_to_pull)}")
        
        except Exception as e:
            logger.error(f"Error syncing collection {class_name} with node {node_id}: {e}")
            raise
    
    def _get_objects_since(self, class_name: str, timestamp: float) -> List[Dict[str, Any]]:
        """Get objects updated since a timestamp"""
        try:
            # Query objects with timestamp filter
            result = self.client.query.get(
                class_name, ["id", "timestamp"]
            ).with_additional(["vector"]).with_where({
                "path": ["timestamp"],
                "operator": "GreaterThan",
                "valueNumber": timestamp
            }).with_limit(self.config["replication"]["batch_size"]).do()
            
            if "data" not in result or "Get" not in result["data"] or class_name not in result["data"]["Get"]:
                return []
            
            return result["data"]["Get"][class_name]
        
        except Exception as e:
            logger.error(f"Error getting objects since {timestamp}: {e}")
            return []
    
    def _get_remote_objects_since(self, remote_client, class_name: str, timestamp: float) -> List[Dict[str, Any]]:
        """Get remote objects updated since a timestamp"""
        try:
            # Query objects with timestamp filter
            result = remote_client.query.get(
                class_name, ["id", "timestamp"]
            ).with_additional(["vector"]).with_where({
                "path": ["timestamp"],
                "operator": "GreaterThan",
                "valueNumber": timestamp
            }).with_limit(self.config["replication"]["batch_size"]).do()
            
            if "data" not in result or "Get" not in result["data"] or class_name not in result["data"]["Get"]:
                return []
            
            return result["data"]["Get"][class_name]
        
        except Exception as e:
            logger.error(f"Error getting remote objects since {timestamp}: {e}")
            return []
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text using the embedding model"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        # Check cache
        import hashlib
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
        
        import hashlib
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
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add knowledge to the database"""
        try:
            # Generate UUID
            object_uuid = str(uuid.uuid4())
            
            # Prepare properties
            properties = {
                "content": content,
                "timestamp": time.time()
            }
            
            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if key in ["source", "topic"]:
                        properties[key] = value
            
            # Add object
            self.client.data_object.create(
                data_object=properties,
                class_name="Knowledge",
                uuid=object_uuid
            )
            
            # Update collection count
            self.collections["Knowledge"]["count"] += 1
            
            return {
                "id": object_uuid,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def add_model_info(self, description: str, model_type: str, model_size: str, parameters: float) -> Dict[str, Any]:
        """Add model information to the database"""
        try:
            # Generate UUID
            object_uuid = str(uuid.uuid4())
            
            # Prepare properties
            properties = {
                "description": description,
                "model_type": model_type,
                "model_size": model_size,
                "parameters": parameters,
                "timestamp": time.time()
            }
            
            # Add object
            self.client.data_object.create(
                data_object=properties,
                class_name="Models",
                uuid=object_uuid
            )
            
            # Update collection count
            self.collections["Models"]["count"] += 1
            
            return {
                "id": object_uuid,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error adding model info: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def add_diagnostic(self, description: str, component: str, status: str, severity: str) -> Dict[str, Any]:
        """Add diagnostic information to the database"""
        try:
            # Generate UUID
            object_uuid = str(uuid.uuid4())
            
            # Prepare properties
            properties = {
                "description": description,
                "component": component,
                "status": status,
                "severity": severity,
                "timestamp": time.time()
            }
            
            # Add object
            self.client.data_object.create(
                data_object=properties,
                class_name="Diagnostics",
                uuid=object_uuid
            )
            
            # Update collection count
            self.collections["Diagnostics"]["count"] += 1
            
            return {
                "id": object_uuid,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error adding diagnostic: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
        def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        try:
            # Get vector for query
            if self.embedding_model:
                # Use our embedding model
                query_vector = self.embed_text(query)
                
                # Search with vector
                result = self.client.query.get(
                    "Knowledge", ["content", "source", "topic", "timestamp"]
                ).with_near_vector({
                    "vector": query_vector
                }).with_limit(limit).do()
            else:
                # Use Weaviate's built-in vectorizer
                result = self.client.query.get(
                    "Knowledge", ["content", "source", "topic", "timestamp"]
                ).with_near_text({
                    "concepts": [query]
                }).with_limit(limit).do()
            
            if "data" not in result or "Get" not in result["data"] or "Knowledge" not in result["data"]["Get"]:
                return []
            
            return result["data"]["Get"]["Knowledge"]
        
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search model information"""
        try:
            # Use Weaviate's built-in vectorizer
            result = self.client.query.get(
                "Models", ["description", "model_type", "model_size", "parameters", "timestamp"]
            ).with_near_text({
                "concepts": [query]
            }).with_limit(limit).do()
            
            if "data" not in result or "Get" not in result["data"] or "Models" not in result["data"]["Get"]:
                return []
            
            return result["data"]["Get"]["Models"]
        
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    def search_diagnostics(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search diagnostic information"""
        try:
            # Use Weaviate's built-in vectorizer
            result = self.client.query.get(
                "Diagnostics", ["description", "component", "status", "severity", "timestamp"]
            ).with_near_text({
                "concepts": [query]
            }).with_limit(limit).do()
            
            if "data" not in result or "Get" not in result["data"] or "Diagnostics" not in result["data"]["Get"]:
                return []
            
            return result["data"]["Get"]["Diagnostics"]
        
        except Exception as e:
            logger.error(f"Error searching diagnostics: {e}")
            return []
    
    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID"""
        try:
            result = self.client.data_object.get_by_id(
                knowledge_id,
                class_name="Knowledge",
                with_vector=True
            )
            
            if not result:
                return None
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting knowledge by ID: {e}")
            return None
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete knowledge by ID"""
        try:
            self.client.data_object.delete(
                uuid=knowledge_id,
                class_name="Knowledge"
            )
            
            # Update collection count
            self.collections["Knowledge"]["count"] -= 1
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")
            return False
    
    def create_class(self, class_name: str, description: str, properties: List[Dict[str, Any]]) -> bool:
        """Create a new class in the schema"""
        try:
            # Check if class already exists
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]
            
            if class_name in existing_classes:
                logger.warning(f"Class {class_name} already exists")
                return False
            
            # Create class object
            class_obj = {
                "class": class_name,
                "description": description,
                "vectorizer": "text2vec-transformers",
                "properties": properties
            }
            
            # Create class
            self.client.schema.create_class(class_obj)
            
            # Store class info
            self.collections[class_name] = {
                "name": class_name,
                "description": description,
                "count": 0
            }
            
            # Update config
            self.config["classes"][class_name] = {
                "description": description,
                "vectorizer": "text2vec-transformers",
                "properties": properties
            }
            self._save_config()
            
            logger.info(f"Created class: {class_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating class: {e}")
            return False
    
    def delete_class(self, class_name: str) -> bool:
        """Delete a class from the schema"""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]
            
            if class_name not in existing_classes:
                logger.warning(f"Class {class_name} does not exist")
                return False
            
            # Delete class
            self.client.schema.delete_class(class_name)
            
            # Remove from collections
            if class_name in self.collections:
                del self.collections[class_name]
            
            # Update config
            if class_name in self.config["classes"]:
                del self.config["classes"][class_name]
                self._save_config()
            
            logger.info(f"Deleted class: {class_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting class: {e}")
            return False
    
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
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the current schema"""
        try:
            return self.client.schema.get()
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return {}
    
    def get_collection_info(self, class_name: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get information about a collection or all collections"""
        if class_name:
            if class_name not in self.collections:
                raise ValueError(f"Class {class_name} not initialized")
            
            # Update count
            self.collections[class_name]["count"] = self._get_collection_count(class_name)
            
            return self.collections[class_name]
        else:
            # Update counts
            for class_name in self.collections:
                self.collections[class_name]["count"] = self._get_collection_count(class_name)
            
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
            "replication_enabled": self.config["replication"]["enabled"],
            "boot_intelligence": self.config["boot_intelligence"]["enabled"]
        }

# Example usage
if __name__ == "__main__":
    # Create Weaviate MCP
    mcp = WeaviateMCP()
    
    # Start MCP
    mcp.start()
    
    # Add some test knowledge
    mcp.add_knowledge(
        content="Weaviate is a vector database that supports multimodal data and GraphQL queries.",
        metadata={"source": "documentation", "topic": "database"}
    )
    
    mcp.add_knowledge(
        content="The RAG MCP uses Weaviate for knowledge storage and retrieval.",
        metadata={"source": "system", "topic": "architecture"}
    )
    
    # Search knowledge
    results = mcp.search_knowledge("How does the vector database work?")
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