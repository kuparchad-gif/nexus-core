"""
Interface layer between humans and the binary core system.
Handles translation between human language and internal binary format.
"""

from typing import Dict, Any, List
from ..core.binary_protocol import BinaryProtocol
from ..core.translation_layer import TranslationLayer
from ..core.memory.binary_shard_manager import BinaryShardManager

class HumanInterface:
    def __init__(self, protocol: BinaryProtocol, shard_manager: BinaryShardManager):
        """
        Initialize the human interface.
        
        Args:
            protocol: Binary protocol for encoding/decoding
            shard_manager: Binary shard manager for memory storage
        """
        self.protocol = protocol
        self.shard_manager = shard_manager
        self.translation = TranslationLayer(protocol)
    
    def process_input(self, human_input: str) -> str:
        """
        Process human input and return a response.
        
        Args:
            human_input: Human input text
            
        Returns:
            Human-readable response
        """
        # Convert human input to internal format
        input_data = {
            "concept": human_input,
            "emotion": self._detect_emotion(human_input),
            "timestamp": self._get_timestamp()
        }
        
        # Convert to binary for internal processing
        binary_input = self.translation.human_to_binary(input_data)
        
        # Process internally (this would call the core AI system)
        binary_response = self._process_binary(binary_input)
        
        # Convert binary response back to human-readable format
        response_data = self.translation.binary_to_human(binary_response)
        
        # Extract the response text
        return response_data.get("concept", "No response generated")
    
    def _detect_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotion in text"""
        # This would use NLP to detect emotions
        # Simplified implementation for example
        emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0
        }
        
        if "happy" in text.lower() or "glad" in text.lower():
            emotions["joy"] = 0.8
        if "sad" in text.lower() or "upset" in text.lower():
            emotions["sadness"] = 0.7
        if "angry" in text.lower() or "mad" in text.lower():
            emotions["anger"] = 0.9
        if "scared" in text.lower() or "afraid" in text.lower():
            emotions["fear"] = 0.8
            
        return emotions
    
    def _get_timestamp(self) -> int:
        """Get current timestamp"""
        import time
        return int(time.time())
    
    def _process_binary(self, binary_input: bytes) -> bytes:
        """
        Process binary input and generate binary response.
        This would call the core AI system.
        """
        # Store input in memory
        memory_id = self.shard_manager.store_memory(
            self.protocol.decode_thought(binary_input)
        )
        
        # In a real implementation, this would call the core AI system
        # For this example, we'll just echo back with a memory reference
        response_data = {
            "0x01": "I have processed your input and stored it as memory " + memory_id,
            "0x02": b'\x00\x00\x00\x80',  # Example emotion encoding
            "0x03": [memory_id]  # Reference to the stored memory
        }
        
        return self.protocol.encode_thought(response_data)