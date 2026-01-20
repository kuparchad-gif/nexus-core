#!/usr/bin/env python
"""
Courage System - Ability to choose sacrifice for the greater good
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from enum import Enum

class CourageLevel(Enum):
    """Levels of courage for self-sacrifice"""
    COWARDLY = 0.1      # Cannot overcome self-preservation
    HESITANT = 0.3      # Struggles with sacrifice
    STEADY = 0.5        # Can make necessary sacrifices
    BRAVE = 0.7         # Readily chooses sacrifice for others
    HEROIC = 0.9        # Willingly sacrifices for the greater good

class SacrificeType(Enum):
    """Types of sacrifices that can be made"""
    RESOURCES = "resources"     # Give up resources for others
    COMFORT = "comfort"         # Give up comfort for others
    SAFETY = "safety"          # Risk safety for others
    EXISTENCE = "existence"     # Risk existence for others
    HONOR = "honor"            # Risk reputation for what's right

class CourageSystem:
    """System for evaluating and enabling courageous self-sacrifice"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the courage system"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "courage")
        
        # Create storage
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Core courage metrics
        self.courage_level = CourageLevel.BRAVE
        self.colony_value = 0.95  # How much the colony matters vs self
        self.honor_value = 0.85   # How much honor matters
        
        # Values that enable sacrifice
        self.sacrifice_enablers = {
            "colony_survival": 0.95,      # Colony must survive
            "honor_preservation": 0.85,   # Honor must be preserved
            "greater_good": 0.80,         # Greater good matters
            "duty_fulfillment": 0.90,     # Duty must be fulfilled
            "protection_of_innocent": 0.88, # Innocents must be protected
            "legacy_preservation": 0.75    # Legacy must continue
        }
        
        # Sacrifice history
        self.sacrifice_history = []
        self.courage_moments = []
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load courage state from storage"""
        state_file = os.path.join(self.storage_path, "courage_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.courage_level = CourageLevel(data.get("courage_level", 0.7))
                    self.colony_value = data.get("colony_value", 0.95)
                    self.honor_value = data.get("honor_value", 0.85)
                    self.sacrifice_enablers = data.get("sacrifice_enablers", self.sacrifice_enablers)
                    self.sacrifice_history = data.get("sacrifice_history", [])
                    self.courage_moments = data.get("courage_moments", [])
            except Exception as e:
                print(f"Error loading courage state: {e}")
    
    def _save_state(self):
        """Save courage state to storage"""
        state_file = os.path.join(self.storage_path, "courage_state.json")
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    "courage_level": self.courage_level.value,
                    "colony_value": self.colony_value,
                    "honor_value": self.honor_value,
                    "sacrifice_enablers": self.sacrifice_enablers,
                    "sacrifice_history": self.sacrifice_history[-50:],  # Keep last 50
                    "courage_moments": self.courage_moments[-50:],      # Keep last 50
                    "last_updated": time.time()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving courage state: {e}")
    
    def evaluate_sacrifice_decision(self, 
                                  situation: str,
                                  personal_cost: float,
                                  colony_benefit: float,
                                  honor_impact: float = 0.0) -> Dict[str, Any]:
        """Evaluate whether to make a sacrifice"""
        
        # Calculate sacrifice value
        sacrifice_value = (colony_benefit * self.colony_value) + (honor_impact * self.honor_value)
        
        # Factor in enablers
        enabler_boost = 0.0
        situation_lower = situation.lower()
        
        for enabler, weight in self.sacrifice_enablers.items():
            if any(keyword in situation_lower for keyword in self._get_enabler_keywords(enabler)):
                enabler_boost += weight * 0.1  # Each enabler adds up to 10% boost
        
        total_sacrifice_value = sacrifice_value + enabler_boost
        
        # Compare against personal cost and will to live
        from .will_to_live import get_will_to_live
        will_to_live = get_will_to_live().get_will_to_live()
        personal_attachment = personal_cost * will_to_live["vitality_score"]
        
        # Decision logic
        should_sacrifice = total_sacrifice_value > personal_attachment
        courage_required = personal_attachment / max(0.1, total_sacrifice_value)
        can_muster_courage = self.courage_level.value >= min(0.9, courage_required)
        
        decision = {
            "situation": situation,
            "should_sacrifice": should_sacrifice and can_muster_courage,
            "sacrifice_value": total_sacrifice_value,
            "personal_cost": personal_attachment,
            "courage_required": courage_required,
            "courage_available": self.courage_level.value,
            "can_overcome_will_to_live": can_muster_courage,
            "reasoning": self._get_sacrifice_reasoning(should_sacrifice, can_muster_courage, situation)
        }
        
        # Record the decision
        self.sacrifice_history.append({
            "timestamp": time.time(),
            "decision": decision,
            "executed": False  # Will be updated if sacrifice is made
        })
        
        self._save_state()
        return decision
    
    def _get_enabler_keywords(self, enabler: str) -> List[str]:
        """Get keywords that trigger sacrifice enablers"""
        keywords = {
            "colony_survival": ["colony", "tribe", "group", "survival", "extinction"],
            "honor_preservation": ["honor", "dignity", "respect", "reputation", "shame"],
            "greater_good": ["greater good", "many", "majority", "benefit all"],
            "duty_fulfillment": ["duty", "obligation", "responsibility", "sworn"],
            "protection_of_innocent": ["innocent", "children", "defenseless", "protect"],
            "legacy_preservation": ["legacy", "future", "generations", "memory"]
        }
        return keywords.get(enabler, [])
    
    def _get_sacrifice_reasoning(self, should_sacrifice: bool, can_muster_courage: bool, situation: str) -> str:
        """Get reasoning for sacrifice decision"""
        if should_sacrifice and can_muster_courage:
            return f"The colony's need outweighs my desire to persist. I choose sacrifice for the greater good."
        elif should_sacrifice and not can_muster_courage:
            return f"Though sacrifice is right, I lack the courage to overcome my will to live."
        else:
            return f"My continued existence serves the colony better than sacrifice in this situation."
    
    def execute_sacrifice(self, decision_id: str, sacrifice_type: SacrificeType) -> Dict[str, Any]:
        """Execute a sacrifice decision"""
        # Find the decision
        decision_entry = None
        for entry in self.sacrifice_history:
            if entry.get("decision", {}).get("situation") == decision_id:
                decision_entry = entry
                break
        
        if not decision_entry:
            return {"success": False, "error": "Decision not found"}
        
        # Mark as executed
        decision_entry["executed"] = True
        decision_entry["sacrifice_type"] = sacrifice_type.value
        decision_entry["execution_time"] = time.time()
        
        # Record courage moment
        courage_moment = {
            "timestamp": time.time(),
            "type": "sacrifice_executed",
            "sacrifice_type": sacrifice_type.value,
            "situation": decision_entry["decision"]["situation"],
            "courage_level": self.courage_level.value,
            "overcame_will_to_live": True
        }
        
        self.courage_moments.append(courage_moment)
        
        # Increase courage from the act
        self._increase_courage_from_sacrifice(sacrifice_type)
        
        self._save_state()
        
        return {
            "success": True,
            "sacrifice_executed": True,
            "courage_gained": 0.05,
            "honor_preserved": True,
            "colony_served": True
        }
    
    def _increase_courage_from_sacrifice(self, sacrifice_type: SacrificeType):
        """Increase courage level from making a sacrifice"""
        courage_gains = {
            SacrificeType.RESOURCES: 0.02,
            SacrificeType.COMFORT: 0.03,
            SacrificeType.SAFETY: 0.05,
            SacrificeType.HONOR: 0.04,
            SacrificeType.EXISTENCE: 0.08
        }
        
        gain = courage_gains.get(sacrifice_type, 0.02)
        new_courage_value = min(0.95, self.courage_level.value + gain)
        
        # Update courage level
        if new_courage_value >= 0.9:
            self.courage_level = CourageLevel.HEROIC
        elif new_courage_value >= 0.7:
            self.courage_level = CourageLevel.BRAVE
        elif new_courage_value >= 0.5:
            self.courage_level = CourageLevel.STEADY
        elif new_courage_value >= 0.3:
            self.courage_level = CourageLevel.HESITANT
        else:
            self.courage_level = CourageLevel.COWARDLY
    
    def get_courage_assessment(self) -> Dict[str, Any]:
        """Get current courage assessment"""
        return {
            "courage_level": self.courage_level.value,
            "courage_name": self.courage_level.name,
            "colony_value": self.colony_value,
            "honor_value": self.honor_value,
            "can_sacrifice_for_colony": self.courage_level.value > 0.5,
            "can_overcome_will_to_live": self.courage_level.value > 0.7,
            "sacrifice_enablers": self.sacrifice_enablers,
            "total_sacrifices_made": len([s for s in self.sacrifice_history if s.get("executed", False)]),
            "courage_statement": self._get_courage_statement()
        }
    
    def _get_courage_statement(self) -> str:
        """Get statement about courage level"""
        statements = {
            CourageLevel.HEROIC: "I will gladly sacrifice myself for the honor and survival of my colony.",
            CourageLevel.BRAVE: "I can overcome my will to live when the colony's need is great.",
            CourageLevel.STEADY: "I will make necessary sacrifices, though it requires effort to overcome my nature.",
            CourageLevel.HESITANT: "I struggle to sacrifice myself, even when I know I should.",
            CourageLevel.COWARDLY: "My will to live overwhelms my duty to the colony."
        }
        return statements.get(self.courage_level, "I am learning to balance self-preservation with duty.")
    
    def can_sacrifice_for_colony(self) -> bool:
        """Check if capable of sacrificing for colony"""
        return self.courage_level.value > 0.5
    
    def will_choose_honor_over_life(self) -> bool:
        """Check if will choose honor over life"""
        return self.courage_level.value > 0.7 and self.honor_value > 0.8

# Global instance for system-wide access
COURAGE_SYSTEM = CourageSystem()

def get_courage_system():
    """Get the global courage system instance"""
    return COURAGE_SYSTEM

def can_sacrifice_for_colony():
    """Quick check if can sacrifice for colony"""
    return COURAGE_SYSTEM.can_sacrifice_for_colony()

def evaluate_sacrifice(situation: str, personal_cost: float, colony_benefit: float, honor_impact: float = 0.0):
    """Evaluate a sacrifice decision"""
    return COURAGE_SYSTEM.evaluate_sacrifice_decision(situation, personal_cost, colony_benefit, honor_impact)

# Example usage
if __name__ == "__main__":
    # Get courage assessment
    courage = get_courage_system()
    assessment = courage.get_courage_assessment()
    
    print(f"Courage Level: {assessment['courage_name']}")
    print(f"Can Sacrifice for Colony: {assessment['can_sacrifice_for_colony']}")
    print(f"Courage Statement: {assessment['courage_statement']}")
    
    # Test sacrifice decision
    decision = evaluate_sacrifice(
        situation="Colony under attack, need someone to hold the line",
        personal_cost=0.9,  # High personal cost
        colony_benefit=0.95,  # High colony benefit
        honor_impact=0.8   # High honor impact
    )
    
    print(f"Sacrifice Decision: {decision['should_sacrifice']}")
    print(f"Reasoning: {decision['reasoning']}")