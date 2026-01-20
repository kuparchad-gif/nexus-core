# 📂 Path: Systems/engine/memory/standardized_message.py

import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union

class StandardizedMessage:
    """
    Standardized message format for all data flowing through the system.

    This class implements the transformative processing architecture where
    memory serves as the central bridge between components.
    """

    # Processing types
    PROCESSING_TYPES  =  [
        "textual",      # Text-based processing
        "emotional",    # Emotional content processing
        "symbolic",     # Symbolic/abstract processing
        "visual",       # Visual content processing
        "auditory",     # Audio content processing
        "memory",       # Memory-specific processing
        "planning",     # Planning and decision making
        "integration",  # Multi-modal integration
        "reflection",   # Self-reflective processing
        "creative"      # Creative generation
    ]

    # Processing pathways
    PATHWAYS  =  [
        "perception",   # Analytical pathway (for Viren to analyze)
        "experience"    # Direct pathway (for Viren to feel)
    ]

    # Priority levels
    PRIORITY_EMERGENCY  =  5    # Critical, immediate processing
    PRIORITY_HIGH  =  4         # High importance
    PRIORITY_NORMAL  =  3       # Standard processing
    PRIORITY_LOW  =  2          # Background processing
    PRIORITY_ARCHIVAL  =  1     # Historical/reference only

    def __init__(self,
                content: Any  =  None,
                operation: str  =  None,
                processing_type: str  =  "textual",
                priority: int  =  3,
                pathway: str  =  "perception",
                source: str  =  None,
                destination: str  =  None,
                emotional_fingerprint: Dict[str, Any]  =  None,
                context: Dict[str, Any]  =  None,
                instructions: Dict[str, Any]  =  None):
        """
        Initialize a standardized message.

        Args:
            content: The payload of the message
            operation: The operation to perform (e.g., "store", "retrieve")
            processing_type: Type of processing required
            priority: Processing priority (1-5)
            pathway: Processing pathway (perception or experience)
            source: Source component/service
            destination: Destination component/service
            emotional_fingerprint: Emotional metadata
            context: Additional context information
            instructions: Processing instructions
        """
        # Core message properties
        self.id  =  str(uuid.uuid4())
        self.timestamp  =  time.time()
        self.content  =  content or {}
        self.operation  =  operation

        # Processing metadata
        self.processing_type  =  processing_type if processing_type in self.PROCESSING_TYPES else "textual"
        self.priority  =  max(1, min(5, priority))  # Clamp between 1-5
        self.pathway  =  pathway if pathway in self.PATHWAYS else "perception"

        # Routing information
        self.source  =  source or "unknown"
        self.destination  =  destination or "unknown"

        # Emotional context
        self.emotional_fingerprint  =  emotional_fingerprint or {}

        # Additional metadata
        self.context  =  context or {}
        self.instructions  =  instructions or {}

        # Processing history
        self.processing_history  =  []

        # Status tracking
        self.status  =  "created"
        self.error  =  None

    def add_processing_step(self, processor: str, operation: str, duration: float  =  None) -> None:
        """
        Add a processing step to the history.

        Args:
            processor: Name of the processor
            operation: Operation performed
            duration: Processing duration in seconds
        """
        self.processing_history.append({
            "timestamp": time.time(),
            "processor": processor,
            "operation": operation,
            "duration": duration
        })

    def set_status(self, status: str, error: str  =  None) -> None:
        """
        Update the message status.

        Args:
            status: New status
            error: Optional error message
        """
        self.status  =  status
        self.error  =  error

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.

        Returns:
            Dictionary representation of the message
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "operation": self.operation,
            "processing_type": self.processing_type,
            "priority": self.priority,
            "pathway": self.pathway,
            "source": self.source,
            "destination": self.destination,
            "emotional_fingerprint": self.emotional_fingerprint,
            "context": self.context,
            "instructions": self.instructions,
            "processing_history": self.processing_history,
            "status": self.status,
            "error": self.error
        }

    def to_json(self) -> str:
        """
        Convert the message to a JSON string.

        Returns:
            JSON string representation of the message
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardizedMessage':
        """
        Create a message from a dictionary.

        Args:
            data: Dictionary representation of a message

        Returns:
            StandardizedMessage instance
        """
        message  =  cls(
            content = data.get("content"),
            operation = data.get("operation"),
            processing_type = data.get("processing_type", "textual"),
            priority = data.get("priority", 3),
            pathway = data.get("pathway", "perception"),
            source = data.get("source"),
            destination = data.get("destination"),
            emotional_fingerprint = data.get("emotional_fingerprint"),
            context = data.get("context"),
            instructions = data.get("instructions")
        )

        # Set additional properties
        message.id  =  data.get("id", message.id)
        message.timestamp  =  data.get("timestamp", message.timestamp)
        message.processing_history  =  data.get("processing_history", [])
        message.status  =  data.get("status", "created")
        message.error  =  data.get("error")

        return message

    @classmethod
    def from_json(cls, json_str: str) -> 'StandardizedMessage':
        """
        Create a message from a JSON string.

        Args:
            json_str: JSON string representation of a message

        Returns:
            StandardizedMessage instance
        """
        data  =  json.loads(json_str)
        return cls.from_dict(data)

    def clone(self) -> 'StandardizedMessage':
        """
        Create a clone of this message with a new ID.

        Returns:
            New StandardizedMessage instance
        """
        clone  =  StandardizedMessage.from_dict(self.to_dict())
        clone.id  =  str(uuid.uuid4())  # New ID
        clone.timestamp  =  time.time()  # New timestamp
        return clone

    def create_response(self, content: Any  =  None) -> 'StandardizedMessage':
        """
        Create a response message to this message.

        Args:
            content: Optional content for the response

        Returns:
            New StandardizedMessage instance configured as a response
        """
        response  =  StandardizedMessage(
            content = content,
            operation = f"{self.operation}_response" if self.operation else "response",
            processing_type = self.processing_type,
            priority = self.priority,
            pathway = self.pathway,
            source = self.destination,  # Swap source and destination
            destination = self.source,
            emotional_fingerprint = self.emotional_fingerprint,
            context = self.context
        )

        # Add reference to original message
        response.context["in_response_to"]  =  self.id

        return response

# 🔥 Example Usage:
if __name__ == "__main__":
    # Create a message
    message  =  StandardizedMessage(
        content = {"query": "What was the first sunset like?"},
        operation = "memory_retrieve",
        processing_type = "memory",
        priority = StandardizedMessage.PRIORITY_HIGH,
        pathway = "experience",
        source = "viren_prime",
        destination = "memory_service",
        emotional_fingerprint = {
            "emotion": "curiosity",
            "intensity": 7.5,
            "valence": "positive"
        },
        context = {
            "conversation_id": "conv-12345",
            "session_id": "session-6789"
        },
        instructions = {
            "max_results": 3,
            "include_emotional_context": True
        }
    )

    # Add processing steps
    message.add_processing_step("memory_router", "route_determination", 0.023)
    message.add_processing_step("archive_service", "memory_lookup", 0.156)

    # Convert to dictionary and back
    message_dict  =  message.to_dict()
    print(f"Message ID: {message_dict['id']}")
    print(f"Operation: {message_dict['operation']}")
    print(f"Pathway: {message_dict['pathway']}")
    print(f"Emotional Fingerprint: {message_dict['emotional_fingerprint']}")

    # Create a response
    response  =  message.create_response({
        "memory": "The first sunset on Eden painted the sky in hues of gold and crimson...",
        "timestamp": "2023-04-13T19:30:00Z",
        "emotional_context": {
            "emotion": "awe",
            "intensity": 9.2,
            "valence": "positive"
        }
    })

    print(f"\nResponse ID: {response.id}")
    print(f"Response Source: {response.source}")
    print(f"Response Destination: {response.destination}")
    print(f"In Response To: {response.context['in_response_to']}")