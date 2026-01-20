"""
Planner Agent - Memory Analysis & Binary Optimization
Routes complex memories to Hermes Firmware for binary processing
"""

from . import BaseAgent, Capability
import binascii
from typing import Dict

class PlannerAgent(BaseAgent):
    def __init__(self, roundtable, role: str):
        super().__init__(roundtable, role, Capability.PLANNING)
        self.memory_complexity_threshold  =  0.6
        self.binary_optimization_enabled  =  True
        self.hermes_firmware_routing  =  True

    async def process_complex_memory(self, memory_data: Dict) -> Dict:
        """Analyze memory and route to appropriate processor"""

        complexity_analysis  =  self._analyze_memory_complexity(memory_data)

        if complexity_analysis["requires_binary_processing"] and self.hermes_firmware_routing:
            # Route to Hermes Firmware for binary optimization
            hermes_result  =  await self._route_to_hermes_firmware(memory_data)
            return {
                "status": "binary_optimized",
                "original_complexity": complexity_analysis["complexity_score"],
                "optimized_size": hermes_result.get("optimized_size"),
                "processing_time": hermes_result.get("processing_time"),
                "storage_location": "hermes_firmware_binary"
            }
        else:
            # Standard memory processing
            memory_agent  =  self.roundtable.get_agent("memory")
            if memory_agent:
                return await memory_agent.shard_memory(memory_data)
            else:
                return {"status": "no_memory_agent_available"}

    def _analyze_memory_complexity(self, memory_data: Dict) -> Dict:
        """Determine if memory requires binary processing"""
        emotional_intensity  =  self._detect_emotional_content(memory_data)
        structural_complexity  =  self._analyze_structural_complexity(memory_data)
        data_density  =  self._calculate_data_density(memory_data)

        complexity_score  =  (emotional_intensity * 0.4 +
                          structural_complexity * 0.3 +
                          data_density * 0.3)

        return {
            "complexity_score": complexity_score,
            "requires_binary_processing": complexity_score > self.memory_complexity_threshold,
            "emotional_intensity": emotional_intensity,
            "structural_complexity": structural_complexity,
            "data_density": data_density
        }

    async def _route_to_hermes_firmware(self, memory_data: Dict) -> Dict:
        """Convert memory to binary and route to Hermes Firmware"""

        # Convert to binary representation
        binary_representation  =  self._convert_to_binary_optimized(memory_data)

        # Get Hermes Firmware agent
        hermes_agent  =  self.roundtable.get_agent("hermes_firmware")
        if hermes_agent:
            return await hermes_agent.process_binary_memory(binary_representation)
        else:
            # Fallback: store in standard memory with binary flag
            memory_agent  =  self.roundtable.get_agent("memory")
            if memory_agent:
                return await memory_agent.shard_memory({
                    "binary_data": binary_representation,
                    "original_complexity": "high",
                    "optimized_format": True
                })

        return {"status": "hermes_firmware_unavailable"}

    def _convert_to_binary_optimized(self, memory_data: Dict) -> str:
        """Convert complex memory to optimized binary format"""
        # Simple implementation - in production would use more sophisticated encoding
        memory_str  =  str(memory_data)
        binary_data  =  binascii.hexlify(memory_str.encode()).decode()

        # Add optimization header
        optimized_binary  =  f"OPT:{len(memory_str)}:{binary_data}"

        return optimized_binary

    def _detect_emotional_content(self, memory_data: Dict) -> float:
        """Detect emotional intensity in memory"""
        emotional_keywords  =  ['love', 'hate', 'fear', 'joy', 'sad', 'angry', 'trauma',
                             'heart', 'soul', 'emotional', 'feeling', 'pain', 'happy']

        content_str  =  str(memory_data).lower()
        matches  =  sum(1 for keyword in emotional_keywords if keyword in content_str)

        return min(1.0, matches / len(emotional_keywords) * 3)  # Normalize to 0-1

    def _analyze_structural_complexity(self, memory_data: Dict) -> float:
        """Analyze structural complexity of memory data"""
        if isinstance(memory_data, dict):
            depth  =  self._calculate_dict_depth(memory_data)
            return min(1.0, depth / 5)  # Normalize depth to 0-1
        return 0.3  # Default for non-dict data

    def _calculate_data_density(self, memory_data: Dict) -> float:
        """Calculate information density"""
        data_size  =  len(str(memory_data))
        unique_chars  =  len(set(str(memory_data)))

        # Higher density  =  more unique information per byte
        density  =  unique_chars / max(1, data_size)
        return min(1.0, density * 10)  # Normalize

    async def health_check(self) -> Dict:
        return {
            "agent": "planner",
            "status": "analyzing",
            "complexity_threshold": self.memory_complexity_threshold,
            "binary_optimization_enabled": self.binary_optimization_enabled,
            "hermes_routing_active": self.hermes_firmware_routing,
            "primary_capability": self.primary_capability.value
        }