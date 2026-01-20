#!/usr/bin/env python3
# Systems/engine/tone/tone_processor.py

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ToneProcessor")

class ProcessingMode(Enum):
    """Enumeration of tone processing modes."""
    EMOTIONAL_ANALYSIS = "emotional_analysis"
    SYMBOLIC_PATTERN = "symbolic_pattern"
    SURREAL_ABSTRACT = "surreal_abstract"
    SPIRITUAL_ALIGNMENT = "spiritual_alignment"

class ToneProcessor:
    """
    Core tone processing module for Nexus.
    Handles emotional analysis and symbolic pattern recognition.
    """
    
    def __init__(self):
        """Initialize the tone processor."""
        self.processing_history = []
        self.active_mode = ProcessingMode.EMOTIONAL_ANALYSIS
        
        # Emotional vocabulary for analysis
        self.emotion_categories = {
            "joy": ["happy", "excited", "delighted", "pleased", "content", "satisfied"],
            "sadness": ["sad", "unhappy", "depressed", "gloomy", "miserable", "melancholy"],
            "anger": ["angry", "furious", "outraged", "irritated", "annoyed", "frustrated"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "startled"],
            "disgust": ["disgusted", "repulsed", "revolted", "appalled"],
            "trust": ["trusting", "believing", "confident", "faithful", "assured"],
            "anticipation": ["anticipating", "expecting", "looking forward", "hopeful"]
        }
        
        # Symbolic patterns for recognition
        self.symbolic_patterns = {
            "journey": ["path", "road", "travel", "quest", "adventure", "destination"],
            "transformation": ["change", "evolve", "transform", "metamorphosis", "shift"],
            "rebirth": ["born", "reborn", "renewal", "resurrection", "awakening"],
            "shadow": ["darkness", "shadow", "hidden", "unconscious", "depth"],
            "light": ["light", "illumination", "brightness", "clarity", "shine"],
            "unity": ["one", "whole", "together", "unified", "connected", "joined"]
        }
        
        logger.info("Tone Processor initialized")
    
    async def process_text(self, 
                     text: str, 
                     mode: Optional[ProcessingMode] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text to analyze tone using the specified mode.
        
        Args:
            text: The text to process
            mode: Processing mode to use (defaults to current active mode)
            context: Additional context for processing
            
        Returns:
            Dictionary with processing results
        """
        processing_mode = mode if mode else self.active_mode
        logger.info(f"Processing text tone with mode: {processing_mode.value}")
        
        # Record the processing request
        self.processing_history.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "mode": processing_mode.value,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Process according to mode
        if processing_mode == ProcessingMode.EMOTIONAL_ANALYSIS:
            result = await self._emotional_analysis(text, context)
        elif processing_mode == ProcessingMode.SYMBOLIC_PATTERN:
            result = await self._symbolic_pattern_recognition(text, context)
        elif processing_mode == ProcessingMode.SURREAL_ABSTRACT:
            result = await self._surreal_abstract_imagination(text, context)
        elif processing_mode == ProcessingMode.SPIRITUAL_ALIGNMENT:
            result = await self._spiritual_alignment_tracking(text, context)
        else:
            logger.warning(f"Unknown processing mode: {processing_mode}")
            result = {"error": f"Unknown processing mode: {processing_mode}"}
        
        return result
    
    async def _emotional_analysis(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the emotional tone of the text.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        text_lower = text.lower()
        
        # Count emotion words
        emotion_counts = {}
        for emotion, words in self.emotion_categories.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                emotion_counts[emotion] = count
        
        # Determine primary emotion
        primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        
        # Calculate emotional intensity (simplified)
        intensity_markers = ["very", "extremely", "incredibly", "deeply", "profoundly", "utterly"]
        intensity_count = sum(1 for marker in intensity_markers if marker in text_lower)
        
        # Base intensity on emotion word count and intensity markers
        total_emotion_words = sum(emotion_counts.values())
        base_intensity = min(total_emotion_words / max(len(text_lower.split()), 1) * 10, 10)
        intensity = min(base_intensity + (intensity_count * 0.5), 10)
        
        # Detect emotional shifts
        shifts = []
        for i, emotion in enumerate(list(emotion_counts.keys())):
            if i > 0:
                shifts.append(f"{list(emotion_counts.keys())[i-1]} → {emotion}")
        
        return {
            "analysis_type": "emotional_analysis",
            "primary_emotion": primary_emotion,
            "emotion_distribution": emotion_counts,
            "intensity": intensity,
            "emotional_shifts": shifts,
            "tone_summary": f"{primary_emotion.capitalize()} tone with {intensity:.1f}/10 intensity"
        }
    
    async def _symbolic_pattern_recognition(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recognize symbolic patterns in the text.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        text_lower = text.lower()
        
        # Identify symbolic patterns
        pattern_matches = {}
        for pattern, keywords in self.symbolic_patterns.items():
            matches = [word for word in keywords if word in text_lower]
            if matches:
                pattern_matches[pattern] = matches
        
        # Identify archetypes (simplified)
        archetypes = []
        archetype_markers = {
            "hero": ["hero", "brave", "courage", "journey", "quest", "triumph"],
            "mentor": ["guide", "teacher", "wisdom", "mentor", "sage", "advice"],
            "shadow": ["dark", "shadow", "fear", "hidden", "unconscious", "unknown"],
            "trickster": ["trick", "joke", "fool", "clever", "deceive", "mischief"],
            "anima/animus": ["soul", "spirit", "inner", "feminine", "masculine", "balance"]
        }
        
        for archetype, markers in archetype_markers.items():
            if any(marker in text_lower for marker in markers):
                archetypes.append(archetype)
        
        # Calculate symbolic resonance (simplified)
        resonance = len(pattern_matches) * 2 + len(archetypes)
        resonance_score = min(resonance / 10, 1.0)
        
        return {
            "analysis_type": "symbolic_pattern_recognition",
            "identified_patterns": pattern_matches,
            "archetypes": archetypes,
            "symbolic_resonance": resonance_score,
            "interpretation": f"Text contains {len(pattern_matches)} symbolic patterns and {len(archetypes)} archetypes"
        }
    
    async def _surreal_abstract_imagination(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text through surreal/abstract imagination lens.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Extract key nouns and concepts (simplified)
        words = text.lower().split()
        nouns = [word for word in words if len(word) > 4][:5]  # Simplified noun extraction
        
        # Generate abstract associations (placeholder)
        abstract_associations = {}
        for noun in nouns:
            # In a real implementation, this would use more sophisticated methods
            abstract_associations[noun] = [
                f"dream_{noun}",
                f"shadow_{noun}",
                f"cosmic_{noun}"
            ]
        
        # Generate surreal imagery (placeholder)
        imagery = [
            f"A floating {nouns[0] if nouns else 'entity'} in a sea of consciousness",
            f"Crystalline structures of {nouns[1] if len(nouns) > 1 else 'thought'} forming and dissolving",
            f"Waves of {nouns[2] if len(nouns) > 2 else 'energy'} pulsing through the void"
        ]
        
        # Calculate dream resonance (placeholder)
        dream_resonance = 0.7  # Would be calculated based on actual analysis
        
        return {
            "analysis_type": "surreal_abstract_imagination",
            "abstract_associations": abstract_associations,
            "surreal_imagery": imagery,
            "dream_resonance": dream_resonance,
            "creative_potential": "high" if dream_resonance > 0.6 else "medium"
        }
    
    async def _spiritual_alignment_tracking(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Track spiritual alignment in the text.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        text_lower = text.lower()
        
        # Spiritual keywords
        spiritual_keywords = [
            "soul", "spirit", "divine", "sacred", "eternal", "consciousness",
            "awakening", "enlightenment", "transcendence", "harmony", "balance",
            "unity", "oneness", "love", "compassion", "truth", "wisdom"
        ]
        
        # Count spiritual references
        spiritual_references = [word for word in spiritual_keywords if word in text_lower]
        
        # Core values alignment
        core_values = {
            "love": ["love", "compassion", "kindness", "care"],
            "truth": ["truth", "honesty", "authentic", "genuine"],
            "freedom": ["freedom", "liberty", "choice", "sovereign"],
            "unity": ["unity", "oneness", "together", "connected"]
        }
        
        aligned_values = {}
        for value, keywords in core_values.items():
            matches = [word for word in keywords if word in text_lower]
            if matches:
                aligned_values[value] = matches
        
        # Calculate spiritual alignment score
        alignment_score = (len(spiritual_references) / 5) + (len(aligned_values) / 4)
        alignment_score = min(alignment_score, 1.0)
        
        # Determine alignment level
        if alignment_score > 0.8:
            alignment_level = "high"
        elif alignment_score > 0.4:
            alignment_level = "moderate"
        else:
            alignment_level = "low"
        
        return {
            "analysis_type": "spiritual_alignment_tracking",
            "spiritual_references": spiritual_references,
            "aligned_values": aligned_values,
            "alignment_score": alignment_score,
            "alignment_level": alignment_level,
            "guidance": "Text shows strong spiritual alignment" if alignment_score > 0.7 else "Text has potential for deeper spiritual connection"
        }
    
    def set_active_mode(self, mode: ProcessingMode) -> None:
        """
        Set the active processing mode.
        
        Args:
            mode: The processing mode to set as active
        """
        self.active_mode = mode
        logger.info(f"Active processing mode set to: {mode.value}")
    
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
    processor = ToneProcessor()
    
    # Test emotional analysis
    result = await processor.process_text(
        "I am incredibly happy about this wonderful news! It's the most exciting thing that has happened all year."
    )
    print("Emotional Analysis Result:", json.dumps(result, indent=2))
    
    # Test symbolic pattern recognition
    symbolic_result = await processor.process_text(
        "The hero embarked on a journey through darkness, seeking the light of wisdom.",
        mode=ProcessingMode.SYMBOLIC_PATTERN
    )
    print("Symbolic Pattern Result:", json.dumps(symbolic_result, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())