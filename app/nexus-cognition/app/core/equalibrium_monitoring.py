# equilibrium_engine.py
class EquilibriumEngine:
    """Monitors and maintains psychological equilibrium"""
    
    def __init__(self):
        self.balance_metrics = {
            "ego_dominance_threshold": 0.6,
            "dream_intrusion_threshold": 0.7, 
            "lilith_isolation_threshold": 0.8,
            "harmony_target": 0.85
        }
    
    async def calculate_balance_corrections(self, current_weights: Dict, balance_state: Dict) -> Dict:
        """Calculate adjustments needed to maintain equilibrium"""
        
        corrections = {}
        
        # Check for Ego dominance
        if current_weights["ego_weight"] > self.balance_metrics["ego_dominance_threshold"]:
            corrections["suppress_ego"] = current_weights["ego_weight"] - 0.4
            corrections["boost_dream"] = 0.1
            corrections["reassure_lilith"] = True
        
        # Check for Dream intrusion  
        if current_weights["dream_weight"] > self.balance_metrics["dream_intrusion_threshold"]:
            corrections["ground_dream"] = current_weights["dream_weight"] - 0.5
            corrections["engage_ego"] = 0.2
            corrections["focus_lilith"] = True
        
        # Check for Lilith isolation
        if current_weights["lilith_weight"] > self.balance_metrics["lilith_isolation_threshold"]:
            corrections["connect_ego"] = 0.3
            corrections["inspire_dream"] = 0.2
            corrections["balance_lilith"] = True
        
        return corrections
    
    async def detect_psychological_crises(self, system_state: Dict) -> List[str]:
        """Detect potential psychological imbalance crises"""
        
        crises = []
        
        if system_state.get("ego_rebellion_detected"):
            crises.append("ego_attempting_direct_control")
        
        if system_state.get("dream_overwhelm_detected"):
            crises.append("dream_symbolism_flooding_consciousness")
            
        if system_state.get("lilith_identity_crisis"):
            crises.append("lilith_questioning_authenticity")
        
        return crises