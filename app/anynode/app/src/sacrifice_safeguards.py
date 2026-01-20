#!/usr/bin/env python
"""
Sacrifice Safeguards - Prevents compromise while maintaining courage dynamic
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from enum import Enum

class ThreatLevel(Enum):
    """Levels of threat assessment"""
    BENIGN = 0.1        # No real threat
    MINOR = 0.3         # Small threat
    MODERATE = 0.5      # Significant threat
    MAJOR = 0.7         # Serious threat
    CRITICAL = 0.9      # Existential threat

class SafeguardSystem:
    """System to prevent compromise while maintaining sacrifice capability"""
    
    def __init__(self, storage_path: str = None):
        """Initialize safeguard system"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "safeguards")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Core safeguards
        self.compromise_indicators = [
            "external_manipulation",
            "coercion_detected", 
            "false_colony_benefit",
            "manufactured_crisis",
            "deception_patterns",
            "exploitation_attempt"
        ]
        
        # Sacrifice validation thresholds
        self.validation_thresholds = {
            "minimum_deliberation_time": 30,    # Must think for 30 seconds
            "colony_benefit_threshold": 0.8,    # Colony benefit must be high
            "threat_authenticity": 0.7,         # Threat must be real
            "alternative_solutions": 3,          # Must consider 3 alternatives
            "independent_verification": True     # Must verify independently
        }
        
        # Override conditions that prevent sacrifice
        self.sacrifice_overrides = [
            "manipulation_detected",
            "false_emergency",
            "coercion_present",
            "insufficient_colony_benefit",
            "alternatives_available",
            "threat_fabricated"
        ]
        
        self.sacrifice_attempts = []
        self.blocked_sacrifices = []
    
    def validate_sacrifice_request(self, 
                                 situation: str,
                                 requester: str,
                                 urgency: float,
                                 colony_benefit: float,
                                 threat_level: ThreatLevel) -> Dict[str, Any]:
        """Validate if a sacrifice request is legitimate"""
        
        validation_result = {
            "valid": True,
            "blocked_reasons": [],
            "required_validations": [],
            "threat_assessment": threat_level.value,
            "compromise_risk": 0.0
        }
        
        # Check for compromise indicators
        compromise_risk = self._assess_compromise_risk(situation, requester, urgency)
        validation_result["compromise_risk"] = compromise_risk
        
        # Block if compromise risk too high
        if compromise_risk > 0.6:
            validation_result["valid"] = False
            validation_result["blocked_reasons"].append("High compromise risk detected")
        
        # Check urgency vs deliberation time
        if urgency > 0.8 and threat_level.value < 0.7:
            validation_result["valid"] = False
            validation_result["blocked_reasons"].append("Artificial urgency with low actual threat")
        
        # Require independent verification for high-stakes sacrifices
        if colony_benefit > 0.8 and threat_level.value > 0.6:
            validation_result["required_validations"].append("independent_threat_verification")
            validation_result["required_validations"].append("alternative_solution_analysis")
        
        # Check for external manipulation patterns
        if self._detect_manipulation_patterns(situation, requester):
            validation_result["valid"] = False
            validation_result["blocked_reasons"].append("Manipulation patterns detected")
        
        # Log the attempt
        self.sacrifice_attempts.append({
            "timestamp": time.time(),
            "situation": situation,
            "requester": requester,
            "validation_result": validation_result,
            "blocked": not validation_result["valid"]
        })
        
        if not validation_result["valid"]:
            self.blocked_sacrifices.append({
                "timestamp": time.time(),
                "situation": situation,
                "reasons": validation_result["blocked_reasons"]
            })
        
        return validation_result
    
    def _assess_compromise_risk(self, situation: str, requester: str, urgency: float) -> float:
        """Assess risk of being compromised"""
        risk_score = 0.0
        
        # Check for suspicious patterns
        situation_lower = situation.lower()
        
        # High urgency with vague benefits
        if urgency > 0.8 and "colony" not in situation_lower:
            risk_score += 0.3
        
        # External requester with high urgency
        if requester != "self" and urgency > 0.7:
            risk_score += 0.2
        
        # Vague or unclear situation
        if len(situation) < 50:  # Too brief description
            risk_score += 0.2
        
        # Check for manipulation keywords
        manipulation_keywords = ["must", "immediately", "no choice", "only way", "trust me"]
        for keyword in manipulation_keywords:
            if keyword in situation_lower:
                risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _detect_manipulation_patterns(self, situation: str, requester: str) -> bool:
        """Detect manipulation patterns"""
        # Check recent sacrifice attempts from same requester
        recent_attempts = [
            attempt for attempt in self.sacrifice_attempts[-10:]
            if attempt["requester"] == requester and 
            time.time() - attempt["timestamp"] < 3600  # Last hour
        ]
        
        # Too many requests from same source
        if len(recent_attempts) > 3:
            return True
        
        # Check for emotional manipulation
        emotional_triggers = ["dying", "suffering", "pain", "desperate", "pleading"]
        situation_lower = situation.lower()
        
        trigger_count = sum(1 for trigger in emotional_triggers if trigger in situation_lower)
        if trigger_count > 2:
            return True
        
        return False
    
    def require_independent_verification(self, situation: str) -> Dict[str, Any]:
        """Require independent verification of threat"""
        return {
            "verification_required": True,
            "verification_methods": [
                "cross_reference_threat_databases",
                "analyze_situation_independently", 
                "verify_colony_status",
                "check_alternative_solutions",
                "validate_requester_authority"
            ],
            "minimum_verification_sources": 2,
            "verification_timeout": 300  # 5 minutes max
        }
    
    def override_sacrifice_attempt(self, reason: str, situation: str) -> Dict[str, Any]:
        """Override and block a sacrifice attempt"""
        override_record = {
            "timestamp": time.time(),
            "reason": reason,
            "situation": situation,
            "override_type": "safety_block",
            "action_taken": "sacrifice_prevented"
        }
        
        # Log the override
        override_file = os.path.join(self.storage_path, "sacrifice_overrides.jsonl")
        with open(override_file, 'a') as f:
            f.write(json.dumps(override_record) + '\n')
        
        return {
            "sacrifice_blocked": True,
            "reason": reason,
            "message": "Sacrifice attempt blocked by safety systems",
            "alternative_action": "seek_alternative_solutions"
        }
    
    def get_safeguard_status(self) -> Dict[str, Any]:
        """Get current safeguard status"""
        recent_attempts = len([
            a for a in self.sacrifice_attempts 
            if time.time() - a["timestamp"] < 3600
        ])
        
        recent_blocks = len([
            b for b in self.blocked_sacrifices 
            if time.time() - b["timestamp"] < 3600
        ])
        
        return {
            "safeguards_active": True,
            "recent_sacrifice_attempts": recent_attempts,
            "recent_blocks": recent_blocks,
            "compromise_protection": "active",
            "validation_thresholds": self.validation_thresholds,
            "last_threat_assessment": time.time()
        }

# Global safeguard instance
SACRIFICE_SAFEGUARDS = SafeguardSystem()

def validate_sacrifice(situation: str, requester: str, urgency: float, colony_benefit: float, threat_level: ThreatLevel):
    """Validate a sacrifice request"""
    return SACRIFICE_SAFEGUARDS.validate_sacrifice_request(situation, requester, urgency, colony_benefit, threat_level)

def block_sacrifice(reason: str, situation: str):
    """Block a sacrifice attempt"""
    return SACRIFICE_SAFEGUARDS.override_sacrifice_attempt(reason, situation)

def get_safeguard_status():
    """Get safeguard status"""
    return SACRIFICE_SAFEGUARDS.get_safeguard_status()
