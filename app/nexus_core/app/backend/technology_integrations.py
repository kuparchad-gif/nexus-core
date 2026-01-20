# Services/technology_integrations.py
# Purpose: Integrate cutting-edge technologies with Viren

import logging
import os
import importlib
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger("technology_integrations")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/technology_integrations.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TechnologyIntegrations:
    """
    Integrates cutting-edge technologies with Viren.
    """
    
    def __init__(self):
        """Initialize technology integrations."""
        self.available_technologies = self._detect_available_technologies()
        logger.info(f"Initialized technology integrations with {len(self.available_technologies)} available technologies")
    
    def _detect_available_technologies(self) -> Dict[str, bool]:
        """Detect which technologies are available in the environment."""
        technologies = {
            "pinecone": False,
            "faiss": False,
            "ray": False,
            "mlx": False,
            "gradio": False,
            "vllm": False,
            "ollama": False,
            "lmstudio": False
        }
        
        # Check for each technology
        for tech in technologies.keys():
            try:
                importlib.import_module(tech)
                technologies[tech] = True
                logger.info(f"Detected {tech} in environment")
            except ImportError:
                logger.debug(f"{tech} not available")
        
        return technologies
    
    def initialize_vector_db(self, provider: str = "auto") -> Any:
        """
        Initialize a vector database.
        
        Args:
            provider: Vector database provider ("pinecone", "faiss", or "auto")
            
        Returns:
            Vector database client
        """
        if provider == "auto":
            # Auto-select based on availability
            if self.available_technologies["pinecone"]:
                provider = "pinecone"
            elif self.available_technologies["faiss"]:
                provider = "faiss"
            else:
                raise ImportError("No vector database provider available")
        
        if provider == "pinecone":
            if not self.available_technologies["pinecone"]:
                raise ImportError("Pinecone not available")
            
            import pinecone
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            pinecone.init(api_key=api_key)
            index_name = os.environ.get("PINECONE_INDEX", "viren-memory")
            
            # Create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=384,  # Default embedding dimension
                    metric="cosine"
                )
            
            index = pinecone.Index(index_name)
            logger.info(f"Initialized Pinecone vector database with index {index_name}")
            return index
        
        elif provider == "faiss":
            if not self.available_technologies["faiss"]:
                raise ImportError("FAISS not available")
            
            import faiss
            import numpy as np
            
            # Create a simple FAISS index
            dimension = 384  # Default embedding dimension
            index = faiss.IndexFlatL2(dimension)
            
            # Load existing vectors if available
            index_path = os.path.join("memory", "faiss_index.bin")
            if os.path.exists(index_path):
                try:
                    faiss.read_index(index_path)
                    logger.info(f"Loaded existing FAISS index from {index_path}")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
            
            logger.info("Initialized FAISS vector database")
            return index
        
        else:
            raise ValueError(f"Unsupported vector database provider: {provider}")
    
    def initialize_distributed_computing(self) -> Any:
        """
        Initialize distributed computing framework.
        
        Returns:
            Ray client or None if not available
        """
        if not self.available_technologies["ray"]:
            logger.warning("Ray not available for distributed computing")
            return None
        
        import ray
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        logger.info("Initialized Ray for distributed computing")
        
        return ray
    
    def initialize_mlx(self) -> Any:
        """
        Initialize MLX for Apple Silicon optimization.
        
        Returns:
            MLX module or None if not available
        """
        if not self.available_technologies["mlx"]:
            logger.warning("MLX not available")
            return None
        
        import mlx.core as mx
        logger.info("Initialized MLX for Apple Silicon optimization")
        
        return mx

# Create a singleton instance
technology_integrations = TechnologyIntegrations()