#!/usr/bin/env python3
"""
RAG MCP Controller for Cloud Viren
Manages the RAG (Retrieval-Augmented Generation) Master Control Program
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGMCPController")

# Import components
try:
    from vector_mcp import VectorMCP
    from model_cascade_manager import ModelCascadeManager
except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    logger.error("Please ensure all component files are in the correct location")
    sys.exit(1)

class RAGMCPController:
    """
    RAG MCP Controller for Cloud Viren
    Manages the RAG (Retrieval-Augmented Generation) Master Control Program
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the RAG MCP Controller"""
        self.config_path = config_path or os.path.join("config", "rag_mcp_config.json")
        self.config = self._load_config()
        self.running = False
        self.status = "initializing"
        
        # Initialize components
        self.vector_mcp = VectorMCP()
        self.model_cascade = ModelCascadeManager()
        
        # RAG-specific settings
        self.retrieval_settings = self.config.get("retrieval", {})
        self.augmentation_settings = self.config.get("augmentation", {})
        self.generation_settings = self.config.get("generation", {})
        
        # Statistics
        self.stats = {
            "retrievals": 0,
            "augmentations": 0,
            "generations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache for RAG results
        self.rag_cache = {}
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        
        logger.info("RAG MCP Controller initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "retrieval": {
                "max_results": 5,
                "min_score": 0.7,
                "collections": ["knowledge", "models", "diagnostics"],
                "filter_threshold": 0.5
            },
            "augmentation": {
                "max_tokens": 2048,
                "include_metadata": True,
                "format": "markdown",
                "summarize_long_docs": True
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
                "stop_sequences": ["\n\n\n"]
            },
            "max_cache_size": 1000,
            "cache_ttl": 3600,  # 1 hour
            "default_model": "3B"
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
                    
                    logger.info("RAG MCP configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading RAG MCP configuration: {e}")
        
        logger.info("Using default RAG MCP configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("RAG MCP configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving RAG MCP configuration: {e}")
            return False
    
    def start(self) -> bool:
        """Start the RAG MCP Controller"""
        if self.running:
            logger.warning("RAG MCP Controller is already running")
            return False
        
        logger.info("Starting RAG MCP Controller")
        self.running = True
        self.status = "starting"
        
        # Start components
        self.vector_mcp.start()
        self.model_cascade.initialize()
        
        self.status = "running"
        logger.info("RAG MCP Controller started successfully")
        return True
    
    def stop(self) -> bool:
        """Stop the RAG MCP Controller"""
        if not self.running:
            logger.warning("RAG MCP Controller is not running")
            return False
        
        logger.info("Stopping RAG MCP Controller")
        self.running = False
        self.status = "stopping"
        
        # Stop components
        self.vector_mcp.stop()
        
        self.status = "stopped"
        logger.info("RAG MCP Controller stopped successfully")
        return True
    
    def process_query(self, query: str, model_size: str = None, 
                     retrieval_options: Dict[str, Any] = None,
                     augmentation_options: Dict[str, Any] = None,
                     generation_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using the RAG pipeline
        
        Args:
            query: The query to process
            model_size: The model size to use (e.g., "3B", "7B")
            retrieval_options: Options for retrieval
            augmentation_options: Options for augmentation
            generation_options: Options for generation
            
        Returns:
            Dictionary with RAG results
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        # Check cache
        cache_key = self._generate_cache_key(query, model_size, 
                                           retrieval_options, 
                                           augmentation_options, 
                                           generation_options)
        
        if cache_key in self.rag_cache:
            self.stats["cache_hits"] += 1
            logger.info(f"Cache hit for query: {query}")
            return self.rag_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Use default model size if not specified
        if not model_size:
            model_size = self.config.get("default_model", "3B")
        
        # Merge options with defaults
        retrieval_opts = self.retrieval_settings.copy()
        if retrieval_options:
            retrieval_opts.update(retrieval_options)
            
        augmentation_opts = self.augmentation_settings.copy()
        if augmentation_options:
            augmentation_opts.update(augmentation_options)
            
        generation_opts = self.generation_settings.copy()
        if generation_options:
            generation_opts.update(generation_options)
        
        try:
            # Step 1: Retrieval
            retrieval_start = time.time()
            retrieved_docs = self._retrieve(query, retrieval_opts)
            retrieval_time = time.time() - retrieval_start
            self.stats["retrievals"] += 1
            
            # Step 2: Augmentation
            augmentation_start = time.time()
            augmented_context = self._augment(query, retrieved_docs, augmentation_opts)
            augmentation_time = time.time() - augmentation_start
            self.stats["augmentations"] += 1
            
            # Step 3: Generation
            generation_start = time.time()
            generated_response = self._generate(query, augmented_context, model_size, generation_opts)
            generation_time = time.time() - generation_start
            self.stats["generations"] += 1
            
            # Prepare result
            result = {
                "query": query,
                "response": generated_response,
                "model_size": model_size,
                "retrieved_docs": retrieved_docs,
                "timing": {
                    "retrieval": retrieval_time,
                    "augmentation": augmentation_time,
                    "generation": generation_time,
                    "total": time.time() - start_time
                },
                "metadata": {
                    "doc_count": len(retrieved_docs),
                    "context_length": len(augmented_context)
                }
            }
            
            # Cache result
            if len(self.rag_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.rag_cache))
                del self.rag_cache[oldest_key]
            
            self.rag_cache[cache_key] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "status": "error",
                "timing": {
                    "total": time.time() - start_time
                }
            }
    
    def _retrieve(self, query: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        logger.info(f"Retrieving documents for query: {query}")
        
        max_results = options.get("max_results", 5)
        min_score = options.get("min_score", 0.7)
        collections = options.get("collections", ["knowledge"])
        
        all_results = []
        
        # Search each collection
        for collection in collections:
            try:
                # Search for relevant documents
                results = self.vector_mcp.search_text(
                    collection_name=collection,
                    query_text=query,
                    limit=max_results,
                    filter=options.get("filter")
                )
                
                # Filter by score
                filtered_results = [r for r in results if r["score"] >= min_score]
                
                # Add collection name to results
                for r in filtered_results:
                    r["collection"] = collection
                
                all_results.extend(filtered_results)
            
            except Exception as e:
                logger.error(f"Error searching collection {collection}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit total results
        return all_results[:max_results]
    
    def _augment(self, query: str, docs: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Augment the query with retrieved documents"""
        logger.info(f"Augmenting query with {len(docs)} documents")
        
        max_tokens = options.get("max_tokens", 2048)
        include_metadata = options.get("include_metadata", True)
        format_type = options.get("format", "markdown")
        summarize_long_docs = options.get("summarize_long_docs", True)
        
        # Build context
        context = f"Query: {query}\n\n"
        context += "Retrieved Information:\n\n"
        
        for i, doc in enumerate(docs):
            # Get document text
            doc_text = doc.get("payload", {}).get("text", "")
            
            # Summarize if too long and option enabled
            if summarize_long_docs and len(doc_text) > 1000:
                doc_text = doc_text[:1000] + "... [truncated]"
            
            # Format based on specified format
            if format_type == "markdown":
                context += f"## Document {i+1} (Score: {doc['score']:.2f})\n\n"
                context += doc_text + "\n\n"
                
                if include_metadata:
                    context += "### Metadata\n\n"
                    for key, value in doc.get("payload", {}).items():
                        if key != "text":
                            context += f"- {key}: {value}\n"
                    context += "\n"
            
            elif format_type == "plain":
                context += f"Document {i+1} (Score: {doc['score']:.2f}):\n"
                context += doc_text + "\n\n"
            
            elif format_type == "json":
                # Just collect data, will format as JSON at the end
                pass
        
        # Ensure context is not too long
        if len(context) > max_tokens * 4:  # Rough estimate of tokens to chars
            context = context[:max_tokens * 4] + "... [context truncated due to length]"
        
        return context
    
    def _generate(self, query: str, context: str, model_size: str, options: Dict[str, Any]) -> str:
        """Generate a response using the model cascade"""
        logger.info(f"Generating response using {model_size} model")
        
        temperature = options.get("temperature", 0.7)
        top_p = options.get("top_p", 0.9)
        max_tokens = options.get("max_tokens", 1024)
        stop_sequences = options.get("stop_sequences", [])
        
        # Get model
        model = self.model_cascade.get_model(model_size=model_size)
        if not model:
            raise ValueError(f"Model size {model_size} not available")
        
        # Build prompt
        prompt = f"{context}\n\nBased on the above information, please answer: {query}\n\nAnswer:"
        
        # In a real implementation, this would call the model
        # For now, we'll simulate a response
        response = f"This is a simulated response from the {model_size} model for query: {query}"
        
        return response
    
    def _generate_cache_key(self, query: str, model_size: str, 
                          retrieval_options: Dict[str, Any],
                          augmentation_options: Dict[str, Any],
                          generation_options: Dict[str, Any]) -> str:
        """Generate a cache key for a query"""
        import hashlib
        
        # Create a string representation of the query and options
        key_str = query
        key_str += f"|model:{model_size or 'default'}"
        
        if retrieval_options:
            key_str += f"|retrieval:{json.dumps(retrieval_options, sort_keys=True)}"
        
        if augmentation_options:
            key_str += f"|augmentation:{json.dumps(augmentation_options, sort_keys=True)}"
        
        if generation_options:
            key_str += f"|generation:{json.dumps(generation_options, sort_keys=True)}"
        
        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def add_knowledge(self, text: str, metadata: Dict[str, Any] = None, 
                     collection: str = "knowledge") -> Dict[str, Any]:
        """Add knowledge to the vector database"""
        try:
            # Generate ID
            import uuid
            doc_id = str(uuid.uuid4())
            
            # Add metadata
            full_metadata = metadata or {}
            full_metadata["added_at"] = time.time()
            
            # Add to vector database
            success = self.vector_mcp.add_text(collection, doc_id, text, full_metadata)
            
            if success:
                return {
                    "id": doc_id,
                    "status": "success",
                    "collection": collection
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to add knowledge"
                }
        
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG MCP statistics"""
        return {
            "status": self.status,
            "running": self.running,
            "stats": self.stats,
            "cache_size": len(self.rag_cache),
            "max_cache_size": self.max_cache_size,
            "vector_mcp": self.vector_mcp.get_status(),
            "model_cascade": self.model_cascade.get_cascade_status()
        }

# Example usage
if __name__ == "__main__":
    # Create RAG MCP Controller
    controller = RAGMCPController()
    
    # Start controller
    controller.start()
    
    # Add some knowledge
    controller.add_knowledge(
        "Cloud Viren is an advanced AI system with a model cascade from 1B to 256B parameters.",
        {"source": "documentation", "topic": "overview"}
    )
    
    controller.add_knowledge(
        "The Vector MCP uses Qdrant as its foundation for vector storage and retrieval.",
        {"source": "documentation", "topic": "architecture"}
    )
    
    controller.add_knowledge(
        "RAG (Retrieval-Augmented Generation) enhances model responses with retrieved information.",
        {"source": "documentation", "topic": "rag"}
    )
    
    # Process a query
    result = controller.process_query("How does Cloud Viren use RAG?")
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Retrieved {len(result['retrieved_docs'])} documents")
    print(f"Timing: {result['timing']}")
    
    # Get stats
    stats = controller.get_stats()
    print(f"Stats: {stats}")
    
    # Keep running for a while
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        controller.stop()
        print("Controller stopped")