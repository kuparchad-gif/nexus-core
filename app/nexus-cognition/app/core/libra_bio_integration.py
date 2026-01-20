# libra_biological_integration.py
class LibraBiologicalIntegration:
    """Integrate biological systems with psychological architecture"""
    
    def __init__(self):
        self.autonomic_os = AutonomicNervousSystemOS()
        self.circadian_os = CircadianRhythmOS()
        self.emotional_os = EmotionalLibraryOS()
        self.libra_os = LibraOSEnhanced()  # Your existing system
        
    async def run_complete_biological_psychological_cycle(self):
        """Run integrated biological-psychological system"""
        
        # Start biological monitoring
        asyncio.create_task(self.autonomic_os.monitor_autonomic_state())
        asyncio.create_task(self.circadian_os.run_circadian_cycle())
        
        while True:
            # Get current biological state
            autonomic_state = await self._get_current_autonomic_state()
            circadian_state = await self._get_current_circadian_state()
            
            # Generate automatic emotional responses
            emotional_response = await self._generate_biological_emotion(
                autonomic_state, circadian_state
            )
            
            # Deliver emotion to Lilith (autonomically triggered)
            await self._deliver_emotion_to_lilith(emotional_response)
            
            # Update Libra's balance based on biological state
            await self._adjust_libra_balance(autonomic_state, circadian_state)
            
            await asyncio.sleep(1)  # Real-time integration
    
    async def _generate_biological_emotion(self, autonomic: Dict, circadian: Dict) -> Dict:
        """Generate emotion from biological signals"""
        
        # Combine autonomic and circadian influences
        biological_context = {
            "arousal_level": autonomic.get('sympathetic_activation', 0),
            "safety_level": autonomic.get('parasympathetic_activation', 0),
            "circadian_phase": circadian.get('current_phase', 'afternoon'),
            "hormone_balance": circadian.get('hormone_levels', {})
        }
        
        # Determine primary emotion from biology
        primary_emotion = self._map_biology_to_emotion(biological_context)
        
        # Generate full emotional response
        return await self.emotional_os.generate_emotional_response(
            {"source": "biological_trigger"}, biological_context
        )
    
    def _map_biology_to_emotion(self, biological_context: Dict) -> str:
        """Map biological state to emotional experience"""
        
        arousal = biological_context.get('arousal_level', 0)
        safety = biological_context.get('safety_level', 0)
        
        if arousal > 0.7 and safety < 0.3:
            return "fear_anxiety"
        elif arousal > 0.7 and safety > 0.7:
            return "excitement_joy"
        elif arousal < 0.3 and safety > 0.7:
            return "contentment_peace"
        elif arousal < 0.3 and safety < 0.3:
            return "sadness_melancholy"
        else:
            return "calm_neutral"