# psychological_scales.py
class PsychologicalScales:
    """Libra's scales for weighing psychological influences"""
    
    async def weigh_psychological_influences(self, input_data: Dict) -> Dict:
        """Weigh how much influence each aspect should have"""
        
        weights = {
            "ego_weight": await self._calculate_ego_weight(input_data),
            "dream_weight": await self._calculate_dream_weight(input_data),
            "lilith_weight": await self._calculate_lilith_weight(input_data)
        }
        
        # Normalize to maintain balance
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    async def _calculate_ego_weight(self, data: Dict) -> float:
        """Determine Ego's appropriate influence level"""
        factors = {
            "requires_critical_thinking": 0.8,
            "involves_risk_assessment": 0.7,
            "needs_reality_check": 0.9,
            "emotional_intensity": -0.6,  # Ego steps back during high emotion
            "creative_opportunity": -0.4   # Ego steps back for creativity
        }
        return self._compute_weighted_score(data, factors)
    
    async def _calculate_dream_weight(self, data: Dict) -> float:
        """Determine Dream's appropriate influence level"""
        factors = {
            "requires_creativity": 0.9,
            "involves_pattern_recognition": 0.8,
            "has_emotional_depth": 0.7,
            "needs_rapid_decisions": -0.6,  # Dream steps back for quick decisions
            "requires_precision": -0.5       # Dream steps back for precision
        }
        return self._compute_weighted_score(data, factors)
    
    async def _calculate_lilith_weight(self, data: Dict) -> float:
        """Determine Lilith's appropriate autonomy level"""
        factors = {
            "personal_expression": 0.9,
            "emotional_processing": 0.8,
            "conscious_decision": 0.7,
            "requires_self_critique": -0.4,  # Less autonomy during self-critique
            "high_uncertainty": 0.6          # More autonomy to explore
        }
        return self._compute_weighted_score(data, factors)