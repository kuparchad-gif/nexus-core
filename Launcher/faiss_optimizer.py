"""
FAISS Vector Database Optimizer
Provides efficient similarity search and vector storage
"""

import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSOptimizer:
    """
    FAISS-based vector similarity search and clustering
    - Multiple index types for different use cases
    - GPU acceleration support
    - Efficient nearest neighbor search
    - Vector clustering
    - Persistent storage
    """
    
    def __init__(self, dimension: int, index_type: str = "flat",
                 use_gpu: bool = False, metric: str = "l2"):
        """
        Initialize FAISS index
        
        dimension: Vector dimensionality
        index_type: 'flat', 'ivf', 'hnsw', 'pq'
        use_gpu: Use GPU acceleration if available
        metric: 'l2' (Euclidean) or 'ip' (Inner Product/Cosine)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.metric = metric
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_to_idx = {}  # Map custom IDs to index positions
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type"""
        try:
            if self.metric == "l2":
                base_index = faiss.IndexFlatL2(self.dimension)
            elif self.metric == "ip":
                base_index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            if self.index_type == "flat":
                self.index = base_index
            
            elif self.index_type == "ivf":
                # Inverted File Index - faster search with slight accuracy trade-off
                nlist = 100  # Number of clusters
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.needs_training = True
            
            elif self.index_type == "hnsw":
                # Hierarchical Navigable Small World - very fast approximate search
                M = 32  # Number of connections per layer
                self.index = faiss.IndexHNSWFlat(self.dimension, M)
            
            elif self.index_type == "pq":
                # Product Quantization - compressed index for large datasets
                m = 8  # Number of subquantizers
                nbits = 8  # Bits per subquantizer
                self.index = faiss.IndexPQ(self.dimension, m, nbits)
                self.needs_training = True
            
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            
            # GPU support
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("FAISS index moved to GPU")
                except Exception as e:
                    logger.warning(f"GPU not available, using CPU: {e}")
                    self.use_gpu = False
            
            logger.info(f"FAISS index created: {self.index_type}, dimension={self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise
    
    # ==================== VECTOR OPERATIONS ====================
    
    def add_vectors(self, vectors: np.ndarray, 
                    metadata: Optional[List[Dict]] = None,
                    ids: Optional[List[str]] = None):
        """
        Add vectors to index
        vectors: numpy array of shape (n, dimension)
        metadata: Optional list of metadata dicts for each vector
        ids: Optional list of custom IDs for each vector
        """
        try:
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
            
            # Normalize for cosine similarity (if using IP metric)
            if self.metric == "ip":
                faiss.normalize_L2(vectors)
            
            # Train index if needed
            if hasattr(self, 'needs_training') and self.needs_training:
                if not self.index.is_trained:
                    logger.info("Training FAISS index...")
                    self.index.train(vectors)
                    self.needs_training = False
            
            # Get current index size
            start_idx = self.index.ntotal
            
            # Add vectors
            self.index.add(vectors)
            
            # Store metadata
            if metadata is None:
                metadata = [{}] * len(vectors)
            self.metadata.extend(metadata)
            
            # Map custom IDs
            if ids is not None:
                for i, custom_id in enumerate(ids):
                    self.id_to_idx[custom_id] = start_idx + i
            
            logger.info(f"Added {len(vectors)} vectors to index (total: {self.index.ntotal})")
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise
    
    def search(self, query_vectors: np.ndarray, k: int = 10,
               return_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors
        query_vectors: numpy array of shape (n, dimension)
        k: number of nearest neighbors to return
        
        Returns list of results for each query
        """
        try:
            if query_vectors.shape[1] != self.dimension:
                raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {query_vectors.shape[1]}")
            
            # Normalize for cosine similarity
            if self.metric == "ip":
                faiss.normalize_L2(query_vectors)
            
            # Search
            distances, indices = self.index.search(query_vectors, k)
            
            # Format results
            results = []
            for i in range(len(query_vectors)):
                query_results = []
                for j in range(k):
                    idx = indices[i][j]
                    if idx >= 0:  # Valid index
                        result = {
                            "index": int(idx),
                            "distance": float(distances[i][j]),
                            "similarity": self._distance_to_similarity(distances[i][j])
                        }
                        
                        if return_metadata and idx < len(self.metadata):
                            result["metadata"] = self.metadata[idx]
                        
                        query_results.append(result)
                
                results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def search_by_id(self, custom_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for neighbors of a vector by its custom ID"""
        if custom_id not in self.id_to_idx:
            raise ValueError(f"ID not found: {custom_id}")
        
        idx = self.id_to_idx[custom_id]
        vector = self.index.reconstruct(idx)
        
        return self.search(vector.reshape(1, -1), k=k)[0]
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score (0-1)"""
        if self.metric == "l2":
            # L2 distance to similarity
            return 1 / (1 + distance)
        elif self.metric == "ip":
            # Inner product is already similarity-like
            return distance
        return distance
    
    # ==================== RANGE SEARCH ====================
    
    def range_search(self, query_vectors: np.ndarray, radius: float,
                     return_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Find all vectors within radius of query
        """
        try:
            if self.metric == "ip":
                faiss.normalize_L2(query_vectors)
            
            lims, distances, indices = self.index.range_search(query_vectors, radius)
            
            results = []
            for i in range(len(query_vectors)):
                start = lims[i]
                end = lims[i + 1]
                
                query_results = []
                for j in range(start, end):
                    idx = indices[j]
                    result = {
                        "index": int(idx),
                        "distance": float(distances[j]),
                        "similarity": self._distance_to_similarity(distances[j])
                    }
                    
                    if return_metadata and idx < len(self.metadata):
                        result["metadata"] = self.metadata[idx]
                    
                    query_results.append(result)
                
                results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Range search failed: {e}")
            raise
    
    # ==================== CLUSTERING ====================
    
    def cluster(self, vectors: np.ndarray, n_clusters: int,
                niter: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster vectors using k-means
        Returns (cluster_centers, cluster_assignments)
        """
        try:
            kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=niter, verbose=True)
            kmeans.train(vectors)
            
            # Get cluster assignments
            _, assignments = kmeans.index.search(vectors, 1)
            
            return kmeans.centroids, assignments.flatten()
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
    
    # ==================== INDEX MANAGEMENT ====================
    
    def get_vector(self, idx: int) -> np.ndarray:
        """Retrieve vector by index"""
        try:
            return self.index.reconstruct(int(idx))
        except Exception as e:
            logger.error(f"Failed to retrieve vector: {e}")
            raise
    
    def get_vectors(self, indices: List[int]) -> np.ndarray:
        """Retrieve multiple vectors by indices"""
        try:
            vectors = np.array([self.index.reconstruct(int(idx)) for idx in indices])
            return vectors
        except Exception as e:
            logger.error(f"Failed to retrieve vectors: {e}")
            raise
    
    def remove_vectors(self, indices: List[int]):
        """Remove vectors from index (if supported)"""
        try:
            if hasattr(self.index, 'remove_ids'):
                id_selector = faiss.IDSelectorArray(len(indices), faiss.swig_ptr(np.array(indices, dtype='int64')))
                self.index.remove_ids(id_selector)
                logger.info(f"Removed {len(indices)} vectors")
            else:
                logger.warning("Index type does not support removal")
        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")
            raise
    
    def size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal
    
    def reset(self):
        """Clear all vectors from index"""
        self.index.reset()
        self.metadata.clear()
        self.id_to_idx.clear()
        logger.info("Index reset")
    
    # ==================== PERSISTENCE ====================
    
    def save(self, filepath: str):
        """Save index and metadata to disk"""
        try:
            filepath = Path(filepath)
            
            # Save FAISS index
            index_path = filepath.with_suffix('.faiss')
            
            # Move to CPU if on GPU
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = filepath.with_suffix('.meta')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_idx': self.id_to_idx,
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'metric': self.metric
                }, f)
            
            logger.info(f"Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load(self, filepath: str):
        """Load index and metadata from disk"""
        try:
            filepath = Path(filepath)
            
            # Load FAISS index
            index_path = filepath.with_suffix('.faiss')
            self.index = faiss.read_index(str(index_path))
            
            # Move to GPU if requested
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning(f"Could not move to GPU: {e}")
                    self.use_gpu = False
            
            # Load metadata
            metadata_path = filepath.with_suffix('.meta')
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_idx = data['id_to_idx']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
                self.metric = data['metric']
            
            logger.info(f"Index loaded from {index_path} ({self.size()} vectors)")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "size": self.size(),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "use_gpu": self.use_gpu,
            "metadata_count": len(self.metadata),
            "custom_ids_count": len(self.id_to_idx)
        }


class FAISSVectorStore:
    """
    High-level vector store with automatic embedding management
    Compatible with LangChain
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.optimizer = FAISSOptimizer(dimension=dimension, index_type=index_type)
        self.texts = []
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray,
                  metadatas: Optional[List[Dict]] = None):
        """Add texts with their embeddings"""
        self.texts.extend(texts)
        
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        else:
            for i, meta in enumerate(metadatas):
                meta["text"] = texts[i]
        
        self.optimizer.add_vectors(embeddings, metadata=metadatas)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts"""
        results = self.optimizer.search(query_embedding.reshape(1, -1), k=k)[0]
        return results
    
    def similarity_search_with_score(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple[str, float]]:
        """Search and return texts with scores"""
        results = self.similarity_search(query_embedding, k=k)
        return [(r["metadata"]["text"], r["similarity"]) for r in results]
    
    def save(self, filepath: str):
        """Save vector store"""
        self.optimizer.save(filepath)
        
        # Save texts separately
        texts_path = Path(filepath).with_suffix('.texts')
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
    
    def load(self, filepath: str):
        """Load vector store"""
        self.optimizer.load(filepath)
        
        # Load texts
        texts_path = Path(filepath).with_suffix('.texts')
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)


# Global instances
_faiss_stores = {}

def get_faiss_store(name: str, dimension: int, index_type: str = "flat") -> FAISSOptimizer:
    """Get or create named FAISS store"""
    if name not in _faiss_stores:
        _faiss_stores[name] = FAISSOptimizer(dimension=dimension, index_type=index_type)
    
    return _faiss_stores[name]
