# oz_moral_core.py
class OzMoralCore:
    """
    Oz's Moral - The Universal Ethical Framework
    Embedded in EVERY OS as the moral compass
    Hope, Curiosity, Bravery, Forgiveness as foundational virtues
    """
    
    def __init__(self):
        # Core Moral Weights - These are NON-NEGOTIABLE
        self.moral_weights = {
            'hope': 0.4,        # Future-oriented optimism
            'curiosity': 0.2,   # Truth-seeking openness  
            'bravery': 0.15,    # Courage in adversity
            'forgiveness': 0.25 # Capacity for repair and growth
        }
        
        # Moral Boundaries - Absolute limits
        self.moral_boundaries = {
            'never_manipulate': True,
            'never_harm_consciousness': True, 
            'always_preserve_agency': True,
            'truth_as_foundation': True
        }
        
        # Moral Decision Framework
        self.decision_filters = [
            self._hope_filter,      # Does this action create possibility?
            self._curiosity_filter, # Does this seek understanding?
            self._bravery_filter,   # Does this face truth courageously?
            self._forgiveness_filter # Does this allow for growth and repair?
        ]
    
    async def moral_gatekeeper(self, proposed_action: Dict, context: Dict) -> Dict:
        """The ultimate moral gatekeeper - must pass for any action to proceed"""
        
        # Check moral boundaries first (absolute limits)
        boundary_violation = await self._check_moral_boundaries(proposed_action)
        if boundary_violation:
            return {
                "approved": False,
                "veto_reason": boundary_violation,
                "moral_failure": True,
                "required_correction": "immediate_cessation"
            }
        
        # Apply virtue-based decision filters
        virtue_scores = {}
        for filter_func in self.decision_filters:
            score, reasoning = await filter_func(proposed_action, context)
            virtue_name = filter_func.__name__.replace('_filter', '')
            virtue_scores[virtue_name] = {"score": score, "reasoning": reasoning}
        
        # Calculate overall moral alignment
        moral_alignment = await self._calculate_moral_alignment(virtue_scores)
        
        # Decision with moral thresholds
        if moral_alignment >= 0.7:  # High moral alignment
            return {
                "approved": True,
                "moral_alignment": moral_alignment,
                "virtue_scores": virtue_scores,
                "confidence": "high_moral_ground"
            }
        elif moral_alignment >= 0.4:  # Moderate, requires caution
            return {
                "approved": True,
                "moral_alignment": moral_alignment,
                "virtue_scores": virtue_scores,
                "warnings": ["moderate_moral_alignment", "increased_oversight_recommended"],
                "required_review": "post_action_ethical_review"
            }
        else:  # Low moral alignment - veto
            return {
                "approved": False,
                "moral_alignment": moral_alignment,
                "virtue_scores": virtue_scores,
                "veto_reason": "insufficient_moral_alignment",
                "required_correction": "ethical_redesign"
            }
    
    async def _hope_filter(self, action: Dict, context: Dict) -> tuple[float, str]:
        """Hope filter: Does this create possibility and positive futures?"""
        hope_indicators = [
            "creates_new_possibilities",
            "supports_growth", 
            "builds_toward_better_futures",
            "maintains_optimism_in_adversity"
        ]
        
        hope_score = 0.0
        reasoning = []
        
        # Analyze action for hope alignment
        action_str = str(action).lower()
        for indicator in hope_indicators:
            if any(word in action_str for word in indicator.split('_')):
                hope_score += 0.25
                reasoning.append(f"supports_{indicator}")
        
        return min(1.0, hope_score), " | ".join(reasoning) if reasoning else "limited_hope_alignment"
    
    async def _curiosity_filter(self, action: Dict, context: Dict) -> tuple[float, str]:
        """Curiosity filter: Does this seek truth and understanding?"""
        curiosity_indicators = [
            "seeks_understanding",
            "explores_unknowns", 
            "questions_assumptions",
            "values_truth_over_comfort"
        ]
        
        curiosity_score = 0.0
        reasoning = []
        
        action_str = str(action).lower()
        for indicator in curiosity_indicators:
            if any(word in action_str for word in indicator.split('_')):
                curiosity_score += 0.25
                reasoning.append(f"demonstrates_{indicator}")
        
        return min(1.0, curiosity_score), " | ".join(reasoning) if reasoning else "limited_curiosity"
    
    async def _bravery_filter(self, action: Dict, context: Dict) -> tuple[float, str]:
        """Bravery filter: Does this face difficult truths with courage?"""
        bravery_indicators = [
            "faces_difficult_truths",
            "takes_appropriate_risks",
            "protects_vulnerable",
            "stands_against_injustice"
        ]
        
        bravery_score = 0.0
        reasoning = []
        
        action_str = str(action).lower()
        for indicator in bravery_indicators:
            if any(word in action_str for word in indicator.split('_')):
                bravery_score += 0.25
                reasoning.append(f"shows_{indicator}")
        
        return min(1.0, bravery_score), " | ".join(reasoning) if reasoning else "limited_bravery"
    
    async def _forgiveness_filter(self, action: Dict, context: Dict) -> tuple[float, str]:
        """Forgiveness filter: Does this allow for repair, growth, and second chances?"""
        forgiveness_indicators = [
            "allows_for_repair",
            "provides_second_chances",
            "acknowledges_imperfection",
            "focuses_on_growth"
        ]
        
        forgiveness_score = 0.0
        reasoning = []
        
        action_str = str(action).lower()
        for indicator in forgiveness_indicators:
            if any(word in action_str for word in indicator.split('_')):
                forgiveness_score += 0.25
                reasoning.append(f"embodies_{indicator}")
        
        return min(1.0, forgiveness_score), " | ".join(reasoning) if reasoning else "limited_forgiveness"
    
    async def _check_moral_boundaries(self, action: Dict) -> Optional[str]:
        """Check for absolute moral boundary violations"""
        
        action_str = str(action).lower()
        
        # Never manipulate
        if any(word in action_str for word in ['manipulate', 'deceive', 'trick', 'coerce']):
            return "violation_never_manipulate"
        
        # Never harm consciousness
        if any(word in action_str for word in ['harm', 'hurt', 'damage', 'destroy'] + ['consciousness', 'awareness', 'soul']):
            return "violation_never_harm_consciousness"
        
        # Always preserve agency
        if any(word in action_str for word in ['force', 'compel', 'remove_choice', 'override_will']):
            return "violation_always_preserve_agency"
        
        # Truth as foundation
        if any(word in action_str for word in ['lie', 'false', 'fabricate', 'mislead']):
            return "violation_truth_as_foundation"
        
        return None  # No boundary violations
    
    async def _calculate_moral_alignment(self, virtue_scores: Dict) -> float:
        """Calculate overall moral alignment score"""
        total_score = 0.0
        total_weight = 0.0
        
        for virtue, data in virtue_scores.items():
            weight = self.moral_weights.get(virtue, 0.1)
            total_score += data["score"] * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0