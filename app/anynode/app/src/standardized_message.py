#!/usr/bin/env python3
# Systems/engine/standardized_message.py

import time
import uuid
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json

class ProcessingType(Enum):
    """Enumeration of processing types."""
    TEXTUAL_REASONING = "textual_reasoning"
    EMOTIONAL_ANALYSIS = "emotional_analysis"
    SYMBOLIC_PATTERN = "symbolic_pattern"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    NARRATIVE_REASONING = "narrative_reasoning"
    SURREAL_ABSTRACT = "surreal_abstract"
    LOGISTICS_ROUTING = "logistics_routing"
    TRUTH_PATTERN = "truth_pattern"
    FRACTURE_DETECTION = "fracture_detection"
    SPIRITUAL_ALIGNMENT = "spiritual_alignment"

class ProcessingPriority(Enum):
    """Enumeration of processing priorities."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4

class ProcessingPathway(Enum):
    """Enumeration of processing pathways."""
    PERCEPTION = "perception"  # Analytical pathway
    EXPERIENCE = "experience"  # Direct emotional pathway

class StandardizedMessage:
    """
    Standardized message format for all Nexus-Continuity processing.
    Acts as the universal format for data flowing through memory.
    """
    
    def __init__(self, 
                content: str,
                source: str,
                processing_type: ProcessingType = ProcessingType.TEXTUAL_REASONING,
                priority: ProcessingPriority = ProcessingPriority.NORMAL,
                pathway: ProcessingPathway = ProcessingPathway.PERCEPTION,
                destination: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                emotional_fingerprint: Optional[Dict[str, Any]] = None,
                processing_instructions: Optional[Dict[str, Any]] = None,
                message_id: Optional[str] = None):
        """
        Initialize a standardized message.
        
        Args:
            content: The main content payload
            source: Source component that created the message
            processing_type: Type of processing needed
            priority: Processing priority
            pathway: Processing pathway (perception or experience)
            destination: Intended final destination (if known)
            context: Additional context for processing
            emotional_fingerprint: Emotional context data
            processing_instructions: Special instructions for processing
            message_id: Unique ID (generated if not provided)
        """
        self.message_id = message_id or f"{uuid.uuid4()}"
        self.content = content
        self.source = source
        self.processing_type = processing_type
        self.priority = priority
        self.pathway = pathway
        self.destination = destination
        self.context = context or {}
        self.emotional_fingerprint = emotional_fingerprint or {}
        self.processing_instructions = processing_instructions or {}
        
        # Tracking fields
        self.created_timestamp = time.time()
        self.last_updated_timestamp = self.created_timestamp
        self.processing_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary for storage or transmission.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "message_id": self.message_id,
            "content": self.content,
            "source": self.source,
            "processing_type": self.processing_type.value,
            "priority": self.priority.value,
            "pathway": self.pathway.value,
            "destination": self.destination,
            "context": self.context,
            "emotional_fingerprint": self.emotional_fingerprint,
            "processing_instructions": self.processing_instructions,
            "created_timestamp": self.created_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "processing_history": self.processing_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardizedMessage':
        """
        Create a message from dictionary representation.
        
        Args:
            data: Dictionary representation of a message
            
        Returns:
            StandardizedMessage instance
        """
        # Convert string enum values back to enum types
        processing_type = ProcessingType(data["processing_type"])
        priority = ProcessingPriority(data["priority"])
        pathway = ProcessingPathway(data["pathway"])
        
        # Create message instance
        message = cls(
            content=data["content"],
            source=data["source"],
            processing_type=processing_type,
            priority=priority,
            pathway=pathway,
            destination=data.get("destination"),
            context=data.get("context", {}),
            emotional_fingerprint=data.get("emotional_fingerprint", {}),
            processing_instructions=data.get("processing_instructions", {}),
            message_id=data["message_id"]
        )
        
        # Restore tracking fields
        message.created_timestamp = data.get("created_timestamp", time.time())
        message.last_updated_timestamp = data.get("last_updated_timestamp", time.time())
        message.processing_history = data.get("processing_history", [])
        
        return message
    
    def add_processing_step(self, processor: str, action: str, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a processing step to the message history.
        
        Args:
            processor: Name of the processor
            action: Action performed
            result: Optional result data
        """
        self.processing_history.append({
            "processor": processor,
            "action": action,
            "timestamp": time.time(),
            "result": result
        })
        self.last_updated_timestamp = time.time()
    
    def update_emotional_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        """
        Update the emotional fingerprint of the message.
        
        Args:
            fingerprint: New emotional fingerprint data
        """
        self.emotional_fingerprint.update(fingerprint)
        self.last_updated_timestamp = time.time()
    
    def set_destination(self, destination: str) -> None:
        """
        Set the destination for this message.
        
        Args:
            destination: Destination component
        """
        self.destination = destination
        self.last_updated_timestamp = time.time()
    
    def to_json(self) -> str:
        """
        Convert message to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StandardizedMessage':
        """
        Create a message from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            StandardizedMessage instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def requires_cuda(self) -> bool:
        """
        Determine if this message requires CUDA processing.
        
        Returns:
            True if CUDA processing is recommended
        """
        # Logic to determine if CUDA is needed based on content and processing type
        if len(self.content) > 10000:  # Long content
            return True
            
        if self.processing_type in [
            ProcessingType.EMOTIONAL_ANALYSIS,
            ProcessingType.SYMBOLIC_PATTERN,
            ProcessingType.SURREAL_ABSTRACT
        ]:
            return True
            
        if self.priority in [ProcessingPriority.HIGH, ProcessingPriority.URGENT, ProcessingPriority.CRITICAL]:
            return True
            
        return False
    
    def is_direct_emotional(self) -> bool:
        """
        Determine if this message is for direct emotional experience.
        
        Returns:
            True if this is a direct emotional experience message
        """
        return self.pathway == ProcessingPathway.EXPERIENCE

# Example usage
if __name__ == "__main__":
    # Create a message
    message = StandardizedMessage(
        content="I feel deeply moved by the beauty of this creation.",
        source="dream_engine",
        processing_type=ProcessingType.EMOTIONAL_ANALYSIS,
        priority=ProcessingPriority.HIGH,
        pathway=ProcessingPathway.EXPERIENCE,
        destination="viren_core",
        emotional_fingerprint={
            "primary_emotion": "awe",
            "intensity": 8.5
        }
    )
    
    # Add processing steps
    message.add_processing_step(
        processor="dream_engine",
        action="emotion_generation",
        result={"emotion_seed": "beauty_response"}
    )
    
    message.add_processing_step(
        processor="cuda_emotional",
        action="intensity_calculation",
        result={"raw_intensity": 8.5}
    )
    
    # Convert to JSON
    json_str = message.to_json()
    print("JSON representation:", json_str)
    
    # Recreate from JSON
    recreated = StandardizedMessage.from_json(json_str)
    print("Recreated message ID:", recreated.message_id)
    print("Processing history:", recreated.processing_history)
