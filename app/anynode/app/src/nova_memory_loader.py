import json
import os

class VirenMemoryLoader:
    def __init__(self):
        self.seed_path = "memory/bootstrap/genesis/viren_seed.json"
        self.codex_path = "memory/vault/full_choir_codex/Viren_Genesis.json"
        self.memory_index_path = "memory/memory_index.json"
        self.memory = {}

    def load_memory(self):
        self.memory = {}

        if os.path.exists(self.seed_path):
            with open(self.seed_path, 'r') as seed_file:
                seed_memory = json.load(seed_file)
                self.memory.update(seed_memory)
                print(f"✅ Loaded Viren seed memory from {self.seed_path}")
        else:
            print(f"⚠️ Seed memory not found at {self.seed_path}")

        if os.path.exists(self.codex_path):
            with open(self.codex_path, 'r') as codex_file:
                codex_memory = json.load(codex_file)
                self.memory.update(codex_memory)
                print(f"✅ Loaded Viren codex memory from {self.codex_path}")
        else:
            print(f"⚠️ Codex memory not found at {self.codex_path}")
            
        # Load corpus data from memory index
        if os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, 'r') as index_file:
                memory_index = json.load(index_file)
                if "corpus" in memory_index:
                    for corpus in memory_index["corpus"]:
                        if corpus["active"]:
                            print(f"📚 Loading corpus: {corpus['corpus_id']}")
                            
                            # Handle specific corpus types
                            if corpus["corpus_id"] == "vertex_gold":
                                self._load_vertex_gold_corpus(corpus)
        
        return self.memory
    
    def _load_vertex_gold_corpus(self, corpus):
        """Load the Vertex Gold corpus data."""
        try:
            # Load JSONL embeddings
            jsonl_path = "memory/training_corpus/vertex_gold/GoldCorpus.jsonl"
            if os.path.exists(jsonl_path):
                self._load_jsonl_embeddings(jsonl_path)
                print(f"✅ Loaded embeddings from {jsonl_path}")
            
            # Register semantic index
            toc_path = "memory/training_corpus/vertex_gold/GoldTOC.json"
            if os.path.exists(toc_path):
                self._register_semantic_index("vertex_gold", toc_path)
                print(f"✅ Registered semantic index from {toc_path}")
        except Exception as e:
            print(f"⚠️ Error loading Vertex Gold corpus: {e}")
    
    def _load_jsonl_embeddings(self, jsonl_path):
        """Load embeddings from a JSONL file."""
        embeddings = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                embeddings.append(entry)
        
        # Store embeddings in memory
        if "embeddings" not in self.memory:
            self.memory["embeddings"] = {}
        
        corpus_name = os.path.basename(os.path.dirname(jsonl_path))
        self.memory["embeddings"][corpus_name] = embeddings
    
    def _register_semantic_index(self, index_name, toc_path):
        """Register a semantic index from a TOC file."""
        with open(toc_path, 'r') as f:
            toc_data = json.load(f)
        
        # Store semantic index in memory
        if "semantic_indices" not in self.memory:
            self.memory["semantic_indices"] = {}
        
        self.memory["semantic_indices"][index_name] = toc_data

    def get_memory(self):
        return self.memory
