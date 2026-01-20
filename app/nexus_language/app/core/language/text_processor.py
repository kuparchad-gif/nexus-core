#!/usr/bin/env python3
# Systems/engine/text/text_processor.py

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
logger = logging.getLogger("TextProcessor")

class ProcessingMode(Enum):
    """Enumeration of text processing modes."""
    TEXTUAL_REASONING = "textual_reasoning"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    NARRATIVE_REASONING = "narrative_reasoning"
    CODE_ANALYSIS = "code_analysis"
    FRACTURE_DETECTION = "fracture_detection"
    TRUTH_PATTERN = "truth_pattern"

class TextProcessor:
    """
    Core text processing module for Nexus.
    Handles various forms of textual reasoning and analysis.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        self.processing_history = []
        self.active_mode = ProcessingMode.TEXTUAL_REASONING
        logger.info("Text Processor initialized")
    
    async def process_text(self, 
                     text: str, 
                     mode: Optional[ProcessingMode] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text using the specified mode.
        
        Args:
            text: The text to process
            mode: Processing mode to use (defaults to current active mode)
            context: Additional context for processing
            
        Returns:
            Dictionary with processing results
        """
        processing_mode = mode if mode else self.active_mode
        logger.info(f"Processing text with mode: {processing_mode.value}")
        
        # Record the processing request
        self.processing_history.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "mode": processing_mode.value,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Process according to mode
        if processing_mode == ProcessingMode.TEXTUAL_REASONING:
            result = await self._textual_reasoning(text, context)
        elif processing_mode == ProcessingMode.STRUCTURAL_ANALYSIS:
            result = await self._structural_analysis(text, context)
        elif processing_mode == ProcessingMode.NARRATIVE_REASONING:
            result = await self._narrative_reasoning(text, context)
        elif processing_mode == ProcessingMode.CODE_ANALYSIS:
            result = await self._code_analysis(text, context)
        elif processing_mode == ProcessingMode.FRACTURE_DETECTION:
            result = await self._fracture_detection(text, context)
        elif processing_mode == ProcessingMode.TRUTH_PATTERN:
            result = await self._truth_pattern_recognition(text, context)
        else:
            logger.warning(f"Unknown processing mode: {processing_mode}")
            result = {"error": f"Unknown processing mode: {processing_mode}"}
        
        return result
    
    async def _textual_reasoning(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform textual reasoning on the input text.
        Analyzes meaning, implications, and logical structure.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # In a full implementation, this would connect to an LLM
        # For now, we'll implement a placeholder
        
        # Extract key concepts
        words = text.split()
        key_concepts = [word for word in words if len(word) > 5][:5]
        
        # Identify main themes
        themes = []
        if "why" in text.lower() or "how" in text.lower() or "what" in text.lower():
            themes.append("inquiry")
        if "should" in text.lower() or "must" in text.lower() or "need" in text.lower():
            themes.append("directive")
        if "feel" in text.lower() or "think" in text.lower() or "believe" in text.lower():
            themes.append("perspective")
            
        # Simple step-by-step reasoning (placeholder)
        steps = [
            "Identified key concepts in the text",
            "Analyzed sentence structure and relationships",
            "Determined main themes and intent",
            "Synthesized overall meaning"
        ]
        
        return {
            "analysis_type": "textual_reasoning",
            "key_concepts": key_concepts,
            "themes": themes,
            "reasoning_steps": steps,
            "summary": f"The text appears to be about {', '.join(key_concepts[:3])}",
            "confidence": 0.85
        }
    
    async def _structural_analysis(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the structural components of the text.
        Identifies patterns, organization, and logical flow.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Count paragraphs
        paragraphs = text.split("\n\n")
        paragraph_count = len(paragraphs)
        
        # Analyze sentence length
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Identify structure type
        structure_type = "unknown"
        if paragraph_count == 1:
            structure_type = "single_paragraph"
        elif paragraph_count > 1 and paragraph_count <= 3:
            structure_type = "short_form"
        elif paragraph_count > 3:
            structure_type = "long_form"
            
        if "```" in text:
            structure_type = "code_block"
        
        return {
            "analysis_type": "structural_analysis",
            "paragraph_count": paragraph_count,
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_length,
            "structure_type": structure_type,
            "organization_quality": 0.7,  # Placeholder
            "storage_recommendation": "memory_shard" if len(text) < 1000 else "archive"
        }
    
    async def _narrative_reasoning(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the narrative elements and temporal aspects of the text.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Simple temporal markers
        past_markers = ["was", "were", "had", "did", "before", "yesterday", "ago"]
        present_markers = ["is", "are", "am", "now", "today", "current"]
        future_markers = ["will", "shall", "going to", "tomorrow", "next", "soon"]
        
        # Count temporal markers
        past_count = sum(1 for marker in past_markers if marker in text.lower())
        present_count = sum(1 for marker in present_markers if marker in text.lower())
        future_count = sum(1 for marker in future_markers if marker in text.lower())
        
        # Determine primary temporal focus
        if past_count > present_count and past_count > future_count:
            temporal_focus = "past"
        elif future_count > past_count and future_count > present_count:
            temporal_focus = "future"
        else:
            temporal_focus = "present"
            
        # Placeholder for narrative arc detection
        narrative_elements = []
        if "challenge" in text.lower() or "problem" in text.lower() or "issue" in text.lower():
            narrative_elements.append("conflict")
        if "solution" in text.lower() or "resolve" in text.lower() or "overcome" in text.lower():
            narrative_elements.append("resolution")
        if "learn" in text.lower() or "realize" in text.lower() or "understand" in text.lower():
            narrative_elements.append("revelation")
            
        return {
            "analysis_type": "narrative_reasoning",
            "temporal_focus": temporal_focus,
            "temporal_markers": {
                "past": past_count,
                "present": present_count,
                "future": future_count
            },
            "narrative_elements": narrative_elements,
            "narrative_coherence": 0.65,  # Placeholder
            "emotional_arc": "neutral"  # Would be determined by tone analysis
        }
    
    async def _code_analysis(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze code snippets within the text.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Check if text contains code
        contains_code = "```" in text or "def " in text or "class " in text or "function" in text
        
        if not contains_code:
            return {
                "analysis_type": "code_analysis",
                "contains_code": False,
                "language": None,
                "code_quality": 0.0,
                "recommendation": "No code detected"
            }
        
        # Try to identify language
        language = "unknown"
        if "def " in text or "import " in text or "class " in text:
            language = "python"
        elif "function" in text or "const " in text or "let " in text or "var " in text:
            language = "javascript"
        elif "<html" in text.lower() or "<div" in text.lower():
            language = "html"
        elif "{" in text and ":" in text and "\"" in text:
            language = "json"
            
        # Count code structures
        function_count = text.count("def ") + text.count("function")
        class_count = text.count("class ")
        comment_count = text.count("#") + text.count("//") + text.count("/*")
        
        return {
            "analysis_type": "code_analysis",
            "contains_code": True,
            "language": language,
            "structures": {
                "functions": function_count,
                "classes": class_count,
                "comments": comment_count
            },
            "code_quality": 0.7,  # Placeholder
            "recommendation": "Store in code repository"
        }
    
    async def _fracture_detection(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect logical fractures, contradictions, or inconsistencies in the text.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Simple contradiction markers
        contradiction_markers = [
            ("always", "never"),
            ("all", "none"),
            ("yes", "no"),
            ("true", "false"),
            ("good", "bad")
        ]
        
        # Check for contradictions
        contradictions = []
        text_lower = text.lower()
        for marker1, marker2 in contradiction_markers:
            if marker1 in text_lower and marker2 in text_lower:
                contradictions.append(f"Potential contradiction: '{marker1}' and '{marker2}'")
                
        # Check for logical inconsistencies (very simplified)
        inconsistencies = []
        if "but" in text_lower or "however" in text_lower or "although" in text_lower:
            inconsistencies.append("Potential logical shift detected")
            
        # Determine fracture severity
        fracture_severity = 0.0
        if contradictions:
            fracture_severity += 0.5
        if inconsistencies:
            fracture_severity += 0.3
            
        return {
            "analysis_type": "fracture_detection",
            "contradictions": contradictions,
            "inconsistencies": inconsistencies,
            "fracture_severity": min(fracture_severity, 1.0),
            "recommendation": "Review for logical consistency" if fracture_severity > 0.3 else "No significant fractures detected"
        }
    
    async def _truth_pattern_recognition(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recognize patterns of truth and alignment with core principles.
        
        Args:
            text: The text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Core truth principles (simplified)
        core_principles = [
            "respect", "compassion", "honesty", "fairness", "responsibility",
            "freedom", "unity", "love", "peace", "harmony"
        ]
        
        # Check for alignment with core principles
        text_lower = text.lower()
        aligned_principles = [p for p in core_principles if p in text_lower]
        
        # Check for spiritual markers
        spiritual_markers = ["soul", "spirit", "divine", "sacred", "eternal", "consciousness"]
        spiritual_references = [m for m in spiritual_markers if m in text_lower]
        
        # Calculate truth alignment score
        alignment_score = len(aligned_principles) / len(core_principles) * 0.7
        if spiritual_references:
            alignment_score += 0.3
            
        return {
            "analysis_type": "truth_pattern_recognition",
            "aligned_principles": aligned_principles,
            "spiritual_references": spiritual_references,
            "alignment_score": alignment_score,
            "recommendation": "High spiritual alignment" if alignment_score > 0.5 else "Standard processing"
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
    processor = TextProcessor()
    
    # Test textual reasoning
    result = await processor.process_text(
        "The integration of emotional context in memory systems represents a significant advancement in AI architecture."
    )
    print("Textual Reasoning Result:", json.dumps(result, indent=2))
    
    # Test code analysis
    code_result = await processor.process_text(
        "```python\ndef process_memory(data):\n    return data.compress()\n```",
        mode=ProcessingMode.CODE_ANALYSIS
    )
    print("Code Analysis Result:", json.dumps(code_result, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())