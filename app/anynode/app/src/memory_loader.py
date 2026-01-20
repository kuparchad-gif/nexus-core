# memory/memory_loader.py
# Purpose: Load and manage memory corpora for Viren

import os
import json
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("memory_loader")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/memory_loader.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MemoryLoader:
    """
    Loads and manages memory corpora for Viren.
    """
    
    def __init__(self, memory_dir: str = None):
        """Initialize the memory loader."""
        self.memory_dir = memory_dir or os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(self.memory_dir, "memory_index.json")
        self.memory_index = self._load_memory_index()
        self.loaded_corpora = {}
        
        logger.info(f"Memory loader initialized with memory directory: {self.memory_dir}")
    
    def _load_memory_index(self) -> Dict[str, Any]:
        """Load the memory index from file."""
        try:
            with open(self.index_path, 'r') as f:
                index = json.load(f)
            logger.info(f"Loaded memory index from {self.index_path}")
            return index
        except Exception as e:
            logger.error(f"Error loading memory index: {e}")
            return {"index": [], "corpus": []}
    
    def load_jsonl_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        Load embeddings from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            Dictionary of loaded embeddings
        """
        embeddings = {}
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if "id" in entry and "embedding" in entry:
                        embeddings[entry["id"]] = {
                            "text": entry.get("text", ""),
                            "embedding": entry["embedding"]
                        }
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {e}")
            return {}
    
    def register_semantic_index(self, corpus_id: str, toc_path: str) -> bool:
        """
        Register a semantic index for a corpus.
        
        Args:
            corpus_id: ID of the corpus
            toc_path: Path to the TOC file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(toc_path, 'r') as f:
                toc = json.load(f)
            
            self.loaded_corpora[corpus_id] = {
                "toc": toc,
                "path": os.path.dirname(toc_path)
            }
            
            logger.info(f"Registered semantic index for corpus {corpus_id} from {toc_path}")
            return True
        except Exception as e:
            logger.error(f"Error registering semantic index for corpus {corpus_id}: {e}")
            return False
    
    def load_all_corpora(self) -> bool:
        """
        Load all corpora defined in the memory index.
        
        Returns:
            True if all corpora were loaded successfully, False otherwise
        """
        success = True
        
        for corpus in self.memory_index.get("corpus", []):
            corpus_id = corpus.get("corpus_id")
            if not corpus_id:
                continue
            
            if not corpus.get("active", True):
                logger.info(f"Skipping inactive corpus: {corpus_id}")
                continue
            
            logger.info(f"Loading corpus: {corpus_id}")
            
            # Handle specific corpus types
            if corpus_id == "vertex_gold":
                jsonl_path = os.path.join(self.memory_dir, "training_corpus", corpus_id, "GoldCorpus.jsonl")
                toc_path = os.path.join(self.memory_dir, "training_corpus", corpus_id, "GoldTOC.json")
                
                embeddings = self.load_jsonl_embeddings(jsonl_path)
                if embeddings:
                    self.loaded_corpora[corpus_id] = {
                        "embeddings": embeddings,
                        "path": os.path.dirname(jsonl_path)
                    }
                    
                    # Register semantic index
                    if not self.register_semantic_index(corpus_id, toc_path):
                        success = False
                else:
                    success = False
            else:
                # Generic corpus loading
                for file in corpus.get("files", []):
                    if file.endswith(".jsonl") and corpus.get("embedding", False):
                        file_path = os.path.join(self.memory_dir, corpus_id, file)
                        embeddings = self.load_jsonl_embeddings(file_path)
                        if embeddings:
                            if corpus_id not in self.loaded_corpora:
                                self.loaded_corpora[corpus_id] = {}
                            self.loaded_corpora[corpus_id]["embeddings"] = embeddings
                        else:
                            success = False
        
        logger.info(f"Loaded {len(self.loaded_corpora)} corpora")
        return success
    
    def process_incoming_folder(self, catalog_path: str) -> bool:
        """
        Process incoming training sets based on a catalog.
        
        Args:
            catalog_path: Path to the memory catalog JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
            
            base_dir = os.path.dirname(catalog_path)
            
            for domain, meta in catalog.get("training_sets", {}).items():
                path = os.path.join(base_dir, "training_sets", domain)
                logger.info(f"Processing training set: {domain} from {path}")
                
                # Process the folder based on metadata
                self._process_folder(path, meta)
            
            return True
        except Exception as e:
            logger.error(f"Error processing incoming folder: {e}")
            return False
    
    def _process_folder(self, folder_path: str, metadata: Dict[str, Any]) -> None:
        """
        Process a folder of training data.
        
        Args:
            folder_path: Path to the folder
            metadata: Metadata about the folder
        """
        # This is a placeholder for the actual processing logic
        logger.info(f"Processing folder {folder_path} with metadata: {metadata}")
        
        # In a real implementation, this would:
        # 1. Parse the files in the folder
        # 2. Extract text and create embeddings
        # 3. Store the embeddings in the appropriate format
        # 4. Update the memory index
        
        # For now, just log that we would process it
        status = metadata.get("status", "unknown")
        intended_use = metadata.get("intended_use", "general")
        logger.info(f"Would process folder with status '{status}' for '{intended_use}'")

# Create a singleton instance
memory_loader = MemoryLoader()

# Example usage
if __name__ == "__main__":
    # Load all corpora
    memory_loader.load_all_corpora()
    
    # Process incoming folder if it exists
    catalog_path = os.path.join(memory_loader.memory_dir, "incoming", "memory_catalog.json")
    if os.path.exists(catalog_path):
        memory_loader.process_incoming_folder(catalog_path)
