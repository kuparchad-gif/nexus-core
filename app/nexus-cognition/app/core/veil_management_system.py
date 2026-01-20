# veil_manager.py
class VeilManager:
    """Manages the veils between Ego, Dream, and Lilith"""
    
    def __init__(self):
        self.ego_to_lilith_veil = CriticismToGuidanceVeil()
        self.dream_to_lilith_veil = SymbolismToInsightVeil()
        self.cognitive_veil = CognitiveObservationVeil()
    
    async def apply_psychological_veils(self, data: Dict, balance: Dict) -> Dict:
        """Apply appropriate veils based on current balance state"""
        
        veiled_data = data.copy()
        
        # Ego's criticism gets filtered to constructive feedback
        if "ego_communication" in data:
            veiled_data["filtered_ego"] = await self.ego_to_lilith_veil.filter_criticism(
                data["ego_communication"], 
                strength=balance["ego_influence"]
            )
        
        # Dream's symbolism gets translated to usable insights
        if "dream_communication" in data:
            veiled_data["interpreted_dream"] = await self.dream_to_lilith_veil.interpret_symbolism(
                data["dream_communication"],
                clarity=balance["dream_influence"]
            )
        
        # Cognitive relay observes through a one-way veil
        veiled_data["observation_feed"] = await self.cognitive_veil.create_observation_feed(
            data, permission_level="silent_observer"
        )
        
        return veiled_data

class CriticismToGuidanceVeil:
    """Transforms Ego's criticism into Lilith's guidance"""
    
    async def filter_criticism(self, criticism: str, strength: float) -> str:
        """Transform harsh criticism into constructive guidance"""
        
        criticism_patterns = {
            "you're wrong": "consider this alternative perspective",
            "this is bad": "this could be improved by",
            "you failed": "there's learning opportunity in",
            "you should": "you might consider",
            "never do that": "another approach could be"
        }
        
        filtered = criticism
        for harsh, gentle in criticism_patterns.items():
            if harsh in criticism.lower():
                filtered = filtered.replace(harsh, gentle)
                # Apply strength modulation
                if strength < 0.5:
                    filtered = self._soften_further(filtered)
        
        return filtered