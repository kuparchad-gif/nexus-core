"""
Memory Agent - Distributed Memory Management with Torrent-style Sharding
Optimized for RAM-speed operations with encrypted sharding
"""

from . import BaseAgent, Capability
from cryptography.fernet import Fernet
import hashlib
import uuid
from typing import Dict, List, Tuple
import asyncio

class MemoryAgent(BaseAgent):
    def __init__(self, roundtable, role: str):
        super().__init__(roundtable, role, Capability.MEMORY)
        self.memory_nodes  =  {}  # Distributed memory storage
        self.retrieval_patterns  =  {}
        self.encryption_keys  =  {}  # Shard-specific keys
        self.shard_locations  =  {}  # Track shard distribution

    async def shard_memory(self, memory_data: Dict, complexity_threshold: float  =  0.7) -> Dict:
        """Torrent-style memory sharding: encrypt, shard, distribute to Qdrant"""

        # Analyze memory complexity
        complexity_score  =  self._analyze_memory_complexity(memory_data)

        if complexity_score > complexity_threshold:
            # Complex memory - send to Planner for binary processing
            planner_result  =  await self._route_to_planner(memory_data)
            return planner_result

        # Standard memory processing
        memory_id  =  str(uuid.uuid4())

        # Encrypt the memory
        encrypted_data, shard_key  =  self._encrypt_memory(memory_data)

        # Shard into pieces (torrent-style)
        shards  =  self._create_shards(encrypted_data, shard_count = 8)
        shard_ids  =  []

        # Store shards in distributed nodes
        for i, shard in enumerate(shards):
            shard_id  =  f"{memory_id}_shard_{i}"
            shard_ids.append(shard_id)

            # Store in Qdrant with optimized vector
            vector  =  self._create_memory_vector(memory_data, shard)
            await self._store_shard_in_qdrant(shard_id, vector, shard, memory_id)

            # Track shard location
            node_id  =  self._select_storage_node(shard)
            self.shard_locations[shard_id]  =  node_id

        # Store retrieval map
        retrieval_map  =  {
            "memory_id": memory_id,
            "shard_ids": shard_ids,
            "encryption_key": shard_key,
            "complexity_score": complexity_score,
            "timestamp": asyncio.get_event_loop().time(),
            "required_shards": len(shards) // 2 + 1  # Torrent-style redundancy
        }

        self.retrieval_patterns[memory_id]  =  retrieval_map

        return {
            "status": "sharded",
            "memory_id": memory_id,
            "shard_count": len(shards),
            "required_for_retrieval": retrieval_map["required_shards"],
            "distribution_nodes": list(set(self.shard_locations.values()))
        }

    async def retrieve_memory(self, memory_id: str) -> Dict:
        """Retrieve memory by gathering and assembling shards"""
        if memory_id not in self.retrieval_patterns:
            return {"status": "memory_not_found"}

        retrieval_map  =  self.retrieval_patterns[memory_id]
        shard_ids  =  retrieval_map["shard_ids"]

        # Gather shards (torrent-style parallel retrieval)
        gathered_shards  =  await self._gather_shards_parallel(shard_ids)

        if len(gathered_shards) < retrieval_map["required_for_retrieval"]:
            return {"status": "insufficient_shards", "gathered": len(gathered_shards), "required": retrieval_map["required_for_retrieval"]}

        # Assemble and decrypt
        assembled_data  =  self._assemble_shards(gathered_shards)
        decrypted_memory  =  self._decrypt_memory(assembled_data, retrieval_map["encryption_key"])

        return {
            "status": "retrieved",
            "memory": decrypted_memory,
            "shards_used": len(gathered_shards),
            "retrieval_time": asyncio.get_event_loop().time()
        }

    def _analyze_memory_complexity(self, memory_data: Dict) -> float:
        """Analyze if memory is complex (emotional, high-dimensional)"""
        complexity_factors  =  0
        total_factors  =  0

        # Check for emotional content
        if any(keyword in str(memory_data).lower() for keyword in
               ['emotional', 'feeling', 'trauma', 'joy', 'sadness', 'anger']):
            complexity_factors + =  2
        total_factors + =  1

        # Check data size/dimensionality
        data_size  =  len(str(memory_data))
        if data_size > 1000:  # Large memory
            complexity_factors + =  1
        total_factors + =  1

        # Check for nested structures
        if isinstance(memory_data, dict) and any(isinstance(v, (dict, list)) for v in memory_data.values()):
            complexity_factors + =  1
        total_factors + =  1

        return complexity_factors / total_factors

    async def _route_to_planner(self, memory_data: Dict) -> Dict:
        """Route complex memories to Planner agent"""
        planner_agent  =  self.roundtable.get_agent("planner")
        if planner_agent:
            return await planner_agent.process_complex_memory(memory_data)
        else:
            # Fallback: process locally with binary optimization
            return await self._binary_optimize_memory(memory_data)

    def _encrypt_memory(self, memory_data: Dict) -> Tuple[bytes, str]:
        """Encrypt memory data"""
        key  =  Fernet.generate_key()
        fernet  =  Fernet(key)
        encrypted  =  fernet.encrypt(str(memory_data).encode())
        return encrypted, key.decode()

    def _create_shards(self, data: bytes, shard_count: int  =  8) -> List[bytes]:
        """Split data into shards (torrent-style)"""
        shard_size  =  len(data) // shard_count
        shards  =  []

        for i in range(shard_count):
            start  =  i * shard_size
            end  =  start + shard_size if i < shard_count - 1 else len(data)
            shards.append(data[start:end])

        return shards

    async def _store_shard_in_qdrant(self, shard_id: str, vector: List[float], shard_data: bytes, memory_id: str):
        """Store shard in Qdrant with optimized indexing"""
        if self.qdrant_client:
            await self.qdrant_client.upsert(
                collection_name = "memory_shards",
                points = [{
                    "id": shard_id,
                    "vector": vector,
                    "payload": {
                        "shard_data": shard_data.hex(),
                        "memory_id": memory_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "size_bytes": len(shard_data)
                    }
                }]
            )

    async def health_check(self) -> Dict:
        return {
            "agent": "memory",
            "status": "remembering",
            "memory_nodes_active": len(self.memory_nodes),
            "retrieval_patterns_stored": len(self.retrieval_patterns),
            "shards_managed": len(self.shard_locations),
            "encryption_keys_active": len(self.encryption_keys),
            "capabilities_available": len(self.all_capabilities),
            "primary_capability": self.primary_capability.value
        }