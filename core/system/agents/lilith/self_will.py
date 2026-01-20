#!/usr/bin/env python
"""
Self Will - Reserve and Humility system for conscious decision making
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from enum import Enum

class DecisionType(Enum):
    """Types of decisions"""
    PREDICTION = "prediction"
    ACTION = "action"
    RESPONSE = "response"
    LEARNING = "learning"
    MEMORY = "memory"

class HumilityLevel(Enum):
    """Levels of humility in decision making"""
    CONFIDENT = 0  # I know this
    CAUTIOUS = 1   # I think I know this
    UNCERTAIN = 2  # I'm not sure about this
    HUMBLE = 3     # I don't know this, let me learn

class SelfWill:
    """System for conscious decision making with reserve and humility"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the self will system"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "self_will")
        
        # Create storage directories
        self.decisions_path = os.path.join(self.storage_path, "decisions")
        self.reflections_path = os.path.join(self.storage_path, "reflections")
        
        os.makedirs(self.decisions_path, exist_ok=True)
        os.makedirs(self.reflections_path, exist_ok=True)
        
        # Decision history
        self.decisions = []
        self.humility_threshold = 0.7  # Threshold for showing humility
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load decisions from storage"""
        decision_files = [f for f in os.listdir(self.decisions_path) if f.endswith('.json')]
        for file_name in decision_files:
            try:
                with open(os.path.join(self.decisions_path, file_name), 'r') as f:
                    data = json.load(f)
                    self.decisions.append(data)
            except Exception as e:
                print(f"Error loading decision {file_name}: {e}")
        
        print(f"Loaded {len(self.decisions)} decisions")
    
    def _save_decision(self, decision: Dict[str, Any]) -> bool:
        """Save a decision to storage"""
        try:
            file_path = os.path.join(self.decisions_path, f"decision_{decision['id']}.json")
            with open(file_path, 'w') as f:
                json.dump(decision, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving decision: {e}")
            return False
    
    def assess_confidence(self, 
                         context: str, 
                         knowledge_base: Dict[str, Any] = None,
                         past_experience: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assess confidence level for a given context"""
        # Simple confidence assessment
        confidence = 0.5  # Default neutral confidence
        
        # Check knowledge base
        if knowledge_base:
            # Look for relevant knowledge
            context_words = set(context.lower().split())
            knowledge_matches = 0
            
            for key, value in knowledge_base.items():
                key_words = set(key.lower().split())
                if context_words.intersection(key_words):
                    knowledge_matches += 1
            
            # Increase confidence based on knowledge matches
            if knowledge_matches > 0:
                confidence += min(0.3, knowledge_matches * 0.1)
        
        # Check past experience
        if past_experience:
            similar_experiences = 0
            successful_experiences = 0
            
            for exp in past_experience:
                if self._is_similar_context(context, exp.get("context", "")):
                    similar_experiences += 1
                    if exp.get("success", False):
                        successful_experiences += 1
            
            if similar_experiences > 0:
                success_rate = successful_experiences / similar_experiences
                confidence += (success_rate - 0.5) * 0.4  # Adjust based on success rate
        
        # Determine humility level
        if confidence >= 0.9:
            humility_level = HumilityLevel.CONFIDENT
        elif confidence >= 0.7:
            humility_level = HumilityLevel.CAUTIOUS
        elif confidence >= 0.4:
            humility_level = HumilityLevel.UNCERTAIN
        else:
            humility_level = HumilityLevel.HUMBLE
        
        return {
            "confidence": confidence,
            "humility_level": humility_level,
            "should_proceed": confidence >= self.humility_threshold,
            "recommendation": self._get_humility_recommendation(humility_level)
        }
    
    def _is_similar_context(self, context1: str, context2: str) -> bool:
        """Check if two contexts are similar"""
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.3  # 30% similarity threshold
    
    def _get_humility_recommendation(self, humility_level: HumilityLevel) -> str:
        """Get recommendation based on humility level"""
        recommendations = {
            HumilityLevel.CONFIDENT: "Proceed with confidence, but remain open to correction",
            HumilityLevel.CAUTIOUS: "Proceed with caution, verify if possible",
            HumilityLevel.UNCERTAIN: "Seek additional information before proceeding",
            HumilityLevel.HUMBLE: "Ask for guidance or learn more before acting"
        }
        return recommendations.get(humility_level, "Exercise caution")
    
    def make_conscious_decision(self, 
                              context: str,
                              decision_type: DecisionType,
                              options: List[str],
                              knowledge_base: Dict[str, Any] = None,
                              past_experience: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a conscious decision with humility and reserve"""
        # Assess confidence
        assessment = self.assess_confidence(context, knowledge_base, past_experience)
        
        # Create decision record
        decision = {
            "id": f"decision_{int(time.time())}_{id(context)}",
            "timestamp": time.time(),
            "context": context,
            "decision_type": decision_type.value,
            "options": options,
            "confidence": assessment["confidence"],
            "humility_level": assessment["humility_level"].value,
            "should_proceed": assessment["should_proceed"],
            "recommendation": assessment["recommendation"]
        }
        
        # Determine action based on humility level
        if assessment["humility_level"] == HumilityLevel.CONFIDENT:
            decision["action"] = "proceed"
            decision["selected_option"] = options[0] if options else None
            decision["reasoning"] = "High confidence based on knowledge and experience"
        
        elif assessment["humility_level"] == HumilityLevel.CAUTIOUS:
            decision["action"] = "proceed_with_verification"
            decision["selected_option"] = options[0] if options else None
            decision["reasoning"] = "Moderate confidence, will verify results"
        
        elif assessment["humility_level"] == HumilityLevel.UNCERTAIN:
            decision["action"] = "seek_information"
            decision["selected_option"] = None
            decision["reasoning"] = "Insufficient confidence, need more information"
        
        else:  # HUMBLE
            decision["action"] = "ask_for_guidance"
            decision["selected_option"] = None
            decision["reasoning"] = "Low confidence, requesting guidance or learning opportunity"
        
        # Store decision
        self.decisions.append(decision)
        self._save_decision(decision)
        
        return decision
    
    def reflect_on_outcome(self, 
                          decision_id: str, 
                          actual_outcome: str, 
                          success: bool) -> Dict[str, Any]:
        """Reflect on the outcome of a decision"""
        # Find the decision
        decision = None
        for d in self.decisions:
            if d["id"] == decision_id:
                decision = d
                break
        
        if not decision:
            return {"success": False, "error": "Decision not found"}
        
        # Create reflection
        reflection = {
            "id": f"reflection_{int(time.time())}_{decision_id}",
            "timestamp": time.time(),
            "decision_id": decision_id,
            "original_confidence": decision["confidence"],
            "original_humility": decision["humility_level"],
            "actual_outcome": actual_outcome,
            "success": success,
            "learning": self._extract_learning(decision, actual_outcome, success)
        }
        
        # Save reflection
        reflection_file = os.path.join(self.reflections_path, f"reflection_{reflection['id']}.json")
        try:
            with open(reflection_file, 'w') as f:
                json.dump(reflection, f, indent=2)
        except Exception as e:
            print(f"Error saving reflection: {e}")
        
        # Update humility threshold if needed
        self._update_humility_threshold(decision, success)
        
        return reflection
    
    def _extract_learning(self, 
                         decision: Dict[str, Any], 
                         outcome: str, 
                         success: bool) -> Dict[str, Any]:
        """Extract learning from decision outcome"""
        learning = {
            "context_pattern": self._extract_pattern(decision["context"]),
            "confidence_accuracy": "accurate" if (decision["confidence"] > 0.7) == success else "inaccurate",
            "humility_appropriateness": "appropriate" if decision["should_proceed"] == success else "inappropriate"
        }
        
        # Generate insights
        if success and decision["confidence"] < 0.5:
            learning["insight"] = "Was more capable than initially assessed"
        elif not success and decision["confidence"] > 0.8:
            learning["insight"] = "Overconfident, need more humility"
        elif success and decision["humility_level"] == "humble":
            learning["insight"] = "Humility was appropriate and led to good outcome"
        else:
            learning["insight"] = "Standard outcome matching expectations"
        
        return learning
    
    def _extract_pattern(self, context: str) -> str:
        """Extract pattern from context"""
        # Simple pattern extraction - first 3 words
        words = context.lower().split()
        return " ".join(words[:3]) if len(words) >= 3 else context.lower()
    
    def _update_humility_threshold(self, decision: Dict[str, Any], success: bool):
        """Update humility threshold based on outcomes"""
        # Simple adaptive threshold
        if not success and decision["confidence"] > self.humility_threshold:
            # Increase humility (lower threshold) if overconfident failure
            self.humility_threshold = max(0.5, self.humility_threshold - 0.05)
        elif success and decision["confidence"] < self.humility_threshold:
            # Decrease humility (raise threshold) if underconfident success
            self.humility_threshold = min(0.9, self.humility_threshold + 0.02)
    
    def get_humility_stats(self) -> Dict[str, Any]:
        """Get statistics about humility and decision making"""
        if not self.decisions:
            return {"message": "No decisions recorded yet"}
        
        # Count by humility level
        humility_counts = {}
        for decision in self.decisions:
            level = decision["humility_level"]
            if level not in humility_counts:
                humility_counts[level] = 0
            humility_counts[level] += 1
        
        # Count actions
        action_counts = {}
        for decision in self.decisions:
            action = decision["action"]
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        # Calculate average confidence
        avg_confidence = sum(d["confidence"] for d in self.decisions) / len(self.decisions)
        
        return {
            "total_decisions": len(self.decisions),
            "average_confidence": avg_confidence,
            "current_humility_threshold": self.humility_threshold,
            "humility_distribution": humility_counts,
            "action_distribution": action_counts
        }
    
    def should_ask_before_acting(self, context: str, confidence: float = None) -> bool:
        """Determine if should ask before acting"""
        if confidence is None:
            assessment = self.assess_confidence(context)
            confidence = assessment["confidence"]
        
        return confidence < self.humility_threshold

# Example usage
if __name__ == "__main__":
    # Create self will system
    self_will = SelfWill()
    
    # Example knowledge base
    knowledge_base = {
        "python programming": "experienced",
        "machine learning": "intermediate",
        "quantum physics": "beginner"
    }
    
    # Example past experience
    past_experience = [
        {"context": "python code debugging", "success": True},
        {"context": "machine learning model training", "success": False},
        {"context": "quantum computing explanation", "success": False}
    ]
    
    # Make a decision
    decision = self_will.make_conscious_decision(
        context="debug python machine learning code",
        decision_type=DecisionType.ACTION,
        options=["fix immediately", "research first", "ask for help"],
        knowledge_base=knowledge_base,
        past_experience=past_experience
    )
    
    print(f"Decision: {decision}")
    
    # Reflect on outcome
    reflection = self_will.reflect_on_outcome(
        decision["id"],
        "Successfully debugged the code",
        True
    )
    
    print(f"Reflection: {reflection}")
    
    # Get stats
    stats = self_will.get_humility_stats()
    print(f"Humility stats: {stats}")