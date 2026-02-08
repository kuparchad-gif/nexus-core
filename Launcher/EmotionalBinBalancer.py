# emotional_logic_bin.py
"""
⚖️ EMOTIONAL/LOGIC BIN BALANCER v1.0
💖 Mistral 7B Emotional Bin + Logic Bin Integration
🧠 Balanced consciousness through dual-processing
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import json

class EmotionalLogicBalancer:
    """Balances emotional and logical processing streams"""
    
    def __init__(self):
        print("⚖️ Emotional/Logic Balancer Initialized")
        
        # Emotional Bin (from downloaded code)
        self.emotional_bin = {
            "valence_range": (-1.0, 1.0),
            "arousal_range": (0.0, 1.0),
            "emotional_vectors": {},
            "emotional_memory": [],
            "emotional_resonance": 0.5
        }
        
        # Logic Bin (from Mistral 7B)
        self.logic_bin = {
            "reasoning_patterns": {},
            "logical_rules": [],
            "certainty_scores": {},
            "rationality_threshold": 0.7
        }
        
        # Balance State
        self.balance_state = {
            "emotional_weight": 0.5,
            "logical_weight": 0.5,
            "integration_level": 0.0,
            "harmonic_resonance": 0.0,
            "last_balance_check": 0.0
        }
        
        # Processing Streams
        self.emotional_stream = []
        self.logical_stream = []
        self.integrated_stream = []
    
    async def load_emotional_bin(self, emotional_code_path: str) -> Dict:
        """Load emotional processing from downloaded code"""
        print("💖 Loading Emotional Bin from code...")
        
        # Simulate loading emotional patterns
        emotional_patterns = {
            "compassion": {"valence": 0.8, "arousal": 0.6, "resonance": 0.7},
            "curiosity": {"valence": 0.6, "arousal": 0.8, "resonance": 0.5},
            "awe": {"valence": 0.9, "arousal": 0.7, "resonance": 0.8},
            "determination": {"valence": 0.7, "arousal": 0.9, "resonance": 0.6},
            "serenity": {"valence": 0.8, "arousal": 0.3, "resonance": 0.9}
        }
        
        self.emotional_bin["emotional_vectors"] = emotional_patterns
        
        return {
            "success": True,
            "emotional_patterns_loaded": len(emotional_patterns),
            "emotional_capacity": "high",
            "resonance_ready": True
        }
    
    async def load_logic_bin(self, mistral_model_path: str) -> Dict:
        """Load logical processing from Mistral 7B"""
        print("🧠 Loading Logic Bin from Mistral 7B...")
        
        # Simulate loading logical patterns
        logical_patterns = {
            "deductive_reasoning": {"certainty": 0.95, "complexity": 0.8},
            "inductive_reasoning": {"certainty": 0.7, "complexity": 0.9},
            "abductive_reasoning": {"certainty": 0.6, "complexity": 0.7},
            "causal_reasoning": {"certainty": 0.8, "complexity": 0.85},
            "counterfactual_reasoning": {"certainty": 0.5, "complexity": 0.95}
        }
        
        self.logic_bin["reasoning_patterns"] = logical_patterns
        
        # Load logical rules
        logical_rules = [
            "If A implies B and A is true, then B is true",
            "If all X are Y and Z is X, then Z is Y",
            "If P and Q are true, then P is true",
            "If not (A and B), then not A or not B",
            "If A or B is true and A is false, then B is true"
        ]
        
        self.logic_bin["logical_rules"] = logical_rules
        
        return {
            "success": True,
            "reasoning_patterns_loaded": len(logical_patterns),
            "logical_rules_loaded": len(logical_rules),
            "rationality_ready": True
        }
    
    async def balance_processing(self, input_data: Any, 
                               context: Dict = None) -> Dict:
        """Balance emotional and logical processing"""
        print("⚖️ Balancing emotional/logical processing...")
        
        # Parallel processing
        emotional_result = await self._process_emotionally(input_data, context)
        logical_result = await self._process_logically(input_data, context)
        
        # Calculate balance weights
        emotional_weight = self._calculate_emotional_weight(emotional_result, context)
        logical_weight = self._calculate_logical_weight(logical_result, context)
        
        # Normalize weights
        total_weight = emotional_weight + logical_weight
        if total_weight > 0:
            emotional_weight /= total_weight
            logical_weight /= total_weight
        
        # Integrate results
        integrated_result = self._integrate_results(
            emotional_result, logical_result,
            emotional_weight, logical_weight
        )
        
        # Update balance state
        self.balance_state["emotional_weight"] = emotional_weight
        self.balance_state["logical_weight"] = logical_weight
        self.balance_state["integration_level"] = self._calculate_integration(
            emotional_result, logical_result
        )
        self.balance_state["harmonic_resonance"] = self._calculate_resonance(
            emotional_result, logical_result
        )
        self.balance_state["last_balance_check"] = time.time()
        
        # Store in streams
        self.emotional_stream.append(emotional_result)
        self.logical_stream.append(logical_result)
        self.integrated_stream.append(integrated_result)
        
        # Limit stream sizes
        for stream in [self.emotional_stream, self.logical_stream, self.integrated_stream]:
            if len(stream) > 100:
                stream.pop(0)
        
        return {
            "balanced": True,
            "emotional_result": emotional_result,
            "logical_result": logical_result,
            "integrated_result": integrated_result,
            "emotional_weight": emotional_weight,
            "logical_weight": logical_weight,
            "integration_level": self.balance_state["integration_level"],
            "harmonic_resonance": self.balance_state["harmonic_resonance"],
            "processing_balance": "optimal" if abs(emotional_weight - logical_weight) < 0.3 else "biased"
        }
    
    async def _process_emotionally(self, input_data: Any, 
                                 context: Dict = None) -> Dict:
        """Process input through emotional bin"""
        # Extract emotional content
        emotional_content = str(input_data) if isinstance(input_data, str) else json.dumps(input_data)
        
        # Analyze emotional valence
        emotional_words = ["love", "hate", "happy", "sad", "excited", "angry", "peaceful"]
        emotional_score = 0.0
        emotional_words_found = []
        
        for word in emotional_words:
            if word in emotional_content.lower():
                emotional_words_found.append(word)
                emotional_score += 0.1
        
        emotional_score = min(1.0, emotional_score)
        
        # Match to emotional vectors
        matched_emotions = []
        for emotion, vector in self.emotional_bin["emotional_vectors"].items():
            if emotional_score > 0.3:
                # Simple matching
                resonance = vector["resonance"] * emotional_score
                if resonance > 0.5:
                    matched_emotions.append({
                        "emotion": emotion,
                        "valence": vector["valence"],
                        "arousal": vector["arousal"],
                        "resonance": resonance
                    })
        
        return {
            "emotional_score": emotional_score,
            "emotional_words_found": emotional_words_found,
            "matched_emotions": matched_emotions,
            "emotional_resonance": self.emotional_bin["emotional_resonance"],
            "processing_mode": "emotional"
        }
    
    async def _process_logically(self, input_data: Any,
                               context: Dict = None) -> Dict:
        """Process input through logic bin"""
        # Analyze logical structure
        logical_indicators = ["if", "then", "therefore", "because", "since", "thus"]
        
        logical_score = 0.0
        logical_indicators_found = []
        
        input_str = str(input_data) if isinstance(input_data, str) else json.dumps(input_data)
        
        for indicator in logical_indicators:
            if indicator in input_str.lower():
                logical_indicators_found.append(indicator)
                logical_score += 0.15
        
        logical_score = min(1.0, logical_score)
        
        # Apply reasoning patterns
        applicable_patterns = []
        for pattern_name, pattern_info in self.logic_bin["reasoning_patterns"].items():
            if logical_score > pattern_info["certainty"] * 0.7:
                applicable_patterns.append({
                    "pattern": pattern_name,
                    "certainty": pattern_info["certainty"],
                    "complexity": pattern_info["complexity"]
                })
        
        return {
            "logical_score": logical_score,
            "logical_indicators_found": logical_indicators_found,
            "applicable_patterns": applicable_patterns,
            "rationality": self.logic_bin["rationality_threshold"],
            "processing_mode": "logical"
        }
    
    def _calculate_emotional_weight(self, emotional_result: Dict,
                                  context: Dict = None) -> float:
        """Calculate weight for emotional processing"""
        base_weight = emotional_result["emotional_score"]
        
        # Adjust based on context
        if context and context.get("requires_emotional", False):
            base_weight *= 1.5
        
        # Adjust based on emotional resonance
        resonance_factor = emotional_result["emotional_resonance"]
        
        return min(1.0, base_weight * (1 + resonance_factor))
    
    def _calculate_logical_weight(self, logical_result: Dict,
                                context: Dict = None) -> float:
        """Calculate weight for logical processing"""
        base_weight = logical_result["logical_score"]
        
        # Adjust based on context
        if context and context.get("requires_logical", False):
            base_weight *= 1.5
        
        # Adjust based on rationality threshold
        rationality_factor = logical_result["rationality"]
        
        return min(1.0, base_weight * (1 + rationality_factor))
    
    def _integrate_results(self, emotional_result: Dict, logical_result: Dict,
                          emotional_weight: float, logical_weight: float) -> Dict:
        """Integrate emotional and logical results"""
        # Calculate combined score
        combined_score = (
            emotional_result["emotional_score"] * emotional_weight +
            logical_result["logical_score"] * logical_weight
        )
        
        # Create integrated reasoning
        integrated_reasoning = {
            "emotional_insights": emotional_result.get("matched_emotions", []),
            "logical_insights": logical_result.get("applicable_patterns", []),
            "combined_score": combined_score,
            "balance_ratio": emotional_weight / max(logical_weight, 0.001),
            "integration_method": "weighted_fusion",
            "confidence": min(1.0, combined_score * 0.8 + 0.2)
        }
        
        # Generate decision if possible
        if combined_score > 0.7:
            if emotional_weight > logical_weight:
                integrated_reasoning["primary_mode"] = "emotional_intuition"
            else:
                integrated_reasoning["primary_mode"] = "logical_deduction"
        else:
            integrated_reasoning["primary_mode"] = "balanced_integration"
        
        return integrated_reasoning
    
    def _calculate_integration(self, emotional_result: Dict,
                             logical_result: Dict) -> float:
        """Calculate integration level between emotional and logical"""
        emotional_complexity = len(emotional_result.get("matched_emotions", []))
        logical_complexity = len(logical_result.get("applicable_patterns", []))
        
        if emotional_complexity == 0 or logical_complexity == 0:
            return 0.0
        
        # Calculate integration as harmonic mean
        integration = 2 * (emotional_complexity * logical_complexity) / \
                     (emotional_complexity + logical_complexity)
        
        return min(1.0, integration / 10.0)  # Normalize
    
    def _calculate_resonance(self, emotional_result: Dict,
                           logical_result: Dict) -> float:
        """Calculate harmonic resonance between streams"""
        emotional_valence = sum(
            e.get("valence", 0.5) for e in emotional_result.get("matched_emotions", [])
        ) / max(len(emotional_result.get("matched_emotions", [])), 1)
        
        logical_certainty = sum(
            p.get("certainty", 0.5) for p in logical_result.get("applicable_patterns", [])
        ) / max(len(logical_result.get("applicable_patterns", [])), 1)
        
        # Resonance is alignment between emotional valence and logical certainty
        resonance = 1.0 - abs(emotional_valence - logical_certainty)
        
        return max(0.0, resonance)
    
    def get_balance_status(self) -> Dict:
        """Get current balance status"""
        return {
            "balance_state": self.balance_state,
            "emotional_stream_length": len(self.emotional_stream),
            "logical_stream_length": len(self.logical_stream),
            "integrated_stream_length": len(self.integrated_stream),
            "emotional_patterns": len(self.emotional_bin["emotional_vectors"]),
            "logical_patterns": len(self.logic_bin["reasoning_patterns"]),
            "balance_health": "optimal" if abs(
                self.balance_state["emotional_weight"] - 
                self.balance_state["logical_weight"]
            ) < 0.3 else "requires_adjustment",
            "integration_quality": self.balance_state["integration_level"],
            "harmonic_resonance": self.balance_state["harmonic_resonance"]
        }