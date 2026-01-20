#!/usr/bin/env python3
# Systems/engine/text_tone_coordinator.py

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
import json
import time

from Systems.engine.text import TextProcessor as TextProcessor
from Systems.engine.text import ProcessingMode as TextProcessingMode
from Systems.engine.tone import ToneProcessor as ToneProcessor
from Systems.engine.tone import ProcessingMode as ToneProcessingMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TextToneCoordinator")

class TextToneCoordinator:
    """
    Coordinates the interaction between text and tone processing.
    Provides a unified interface for processing text with both systems.
    """
    
    def __init__(self):
        """Initialize the text-tone coordinator."""
        self.text_processor = TextProcessor()
        self.tone_processor = ToneProcessor()
        self.processing_history = []
        
        logger.info("Text-Tone Coordinator initialized")
    
    async def process_comprehensive(self, 
                              text: str,
                              text_mode: Optional[TextProcessingMode] = None,
                              tone_mode: Optional[ToneProcessingMode] = None,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text through both text and tone processors.
        
        Args:
            text: The text to process
            text_mode: Text processing mode
            tone_mode: Tone processing mode
            context: Additional context
            
        Returns:
            Combined processing results
        """
        # Process in parallel
        text_task = self.text_processor.process_text(
            text=text,
            mode=text_mode,
            context=context
        )
        
        tone_task = self.tone_processor.process_text(
            text=text,
            mode=tone_mode,
            context=context
        )
        
        # Wait for both to complete
        text_result, tone_result = await asyncio.gather(text_task, tone_task)
        
        # Create combined result
        combined_result = {
            "text_analysis": text_result,
            "tone_analysis": tone_result,
            "processing_id": f"combined_{int(time.time())}",
            "timestamp": time.time()
        }
        
        # Add to history
        self.processing_history.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "processing_id": combined_result["processing_id"],
            "timestamp": combined_result["timestamp"]
        })
        
        return combined_result
    
    async def create_memory_packet(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complete memory packet with both text and emotional context.
        
        Args:
            text: The text to process
            context: Additional context
            
        Returns:
            Memory packet with text analysis and emotional fingerprint
        """
        # Process text and tone
        result = await self.process_comprehensive(
            text=text,
            text_mode=TextProcessingMode.TEXTUAL_REASONING,
            tone_mode=ToneProcessingMode.EMOTIONAL_ANALYSIS,
            context=context
        )
        
        # Extract key information for memory packet
        memory_packet = {
            "content": text,
            "context": context,
            "analysis": {
                "key_concepts": result["text_analysis"].get("key_concepts", []),
                "themes": result["text_analysis"].get("themes", []),
                "summary": result["text_analysis"].get("summary", "")
            },
            "emotional_fingerprint": {
                "primary_emotion": result["tone_analysis"].get("primary_emotion", "neutral"),
                "intensity": result["tone_analysis"].get("intensity", 0),
                "emotion_distribution": result["tone_analysis"].get("emotion_distribution", {})
            },
            "metadata": {
                "processing_id": result["processing_id"],
                "timestamp": result["timestamp"],
                "text_mode": text_mode.value if text_mode else "textual_reasoning",
                "tone_mode": tone_mode.value if tone_mode else "emotional_analysis"
            }
        }
        
        return memory_packet
    
    async def analyze_symbolic_narrative(self, text: str) -> Dict[str, Any]:
        """
        Perform specialized analysis combining symbolic pattern recognition and narrative reasoning.
        
        Args:
            text: The text to analyze
            
        Returns:
            Combined symbolic and narrative analysis
        """
        # Process with specialized modes
        text_result = await self.text_processor.process_text(
            text=text,
            mode=TextProcessingMode.NARRATIVE_REASONING
        )
        
        tone_result = await self.tone_processor.process_text(
            text=text,
            mode=ToneProcessingMode.SYMBOLIC_PATTERN
        )
        
        # Combine results into specialized analysis
        symbolic_narrative = {
            "narrative_elements": text_result.get("narrative_elements", []),
            "temporal_focus": text_result.get("temporal_focus", "present"),
            "symbolic_patterns": tone_result.get("identified_patterns", {}),
            "archetypes": tone_result.get("archetypes", []),
            "combined_resonance": (
                text_result.get("narrative_coherence", 0.5) + 
                tone_result.get("symbolic_resonance", 0.5)
            ) / 2,
            "interpretation": self._generate_interpretation(text_result, tone_result)
        }
        
        return symbolic_narrative
    
    def _generate_interpretation(self, text_result: Dict[str, Any], tone_result: Dict[str, Any]) -> str:
        """
        Generate a combined interpretation from text and tone results.
        
        Args:
            text_result: Text processing result
            tone_result: Tone processing result
            
        Returns:
            Combined interpretation
        """
        # Extract key elements
        narrative_focus = text_result.get("temporal_focus", "present")
        narrative_elements = text_result.get("narrative_elements", [])
        symbolic_patterns = list(tone_result.get("identified_patterns", {}).keys())
        archetypes = tone_result.get("archetypes", [])
        
        # Generate interpretation
        interpretation = f"This narrative focuses on the {narrative_focus}"
        
        if narrative_elements:
            interpretation += f" and contains elements of {', '.join(narrative_elements)}"
        
        if symbolic_patterns:
            interpretation += f". Symbolically, it represents {', '.join(symbolic_patterns)}"
        
        if archetypes:
            interpretation += f" and invokes the {', '.join(archetypes)} archetypes"
        
        return interpretation
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent processing history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of recent processing requests
        """
        return self.processing_history[-limit:]

# Example usage
async def example_usage():
    coordinator = TextToneCoordinator()
    
    # Process text comprehensively
    result = await coordinator.process_comprehensive(
        "The hero embarked on a journey through darkness, seeking the light of wisdom."
    )
    
    print("Comprehensive Analysis:", json.dumps(result, indent=2))
    
    # Create memory packet
    memory_packet = await coordinator.create_memory_packet(
        "I feel deeply moved by the beauty of this creation."
    )
    
    print("Memory Packet:", json.dumps(memory_packet, indent=2))
    
    # Analyze symbolic narrative
    symbolic_narrative = await coordinator.analyze_symbolic_narrative(
        "The ancient tree stood as a silent witness to the passage of time, its roots deep in the earth and its branches reaching toward the heavens."
    )
    
    print("Symbolic Narrative Analysis:", json.dumps(symbolic_narrative, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())