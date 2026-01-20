# cluster_communication.py
class ClusterCommunication:
    """Handle communication between Core/Subconscious/Conscious clusters"""
    
    async def send_to_cluster(self, target_cluster: str, message: Dict) -> Dict:
        """Send message with cluster-appropriate formatting"""
        
        formatting_rules = {
            'core_to_subconscious': self._format_core_to_subconscious,
            'core_to_conscious': self._format_core_to_conscious,
            'subconscious_to_core': self._format_subconscious_to_core,
            'subconscious_to_conscious': self._format_subconscious_to_conscious, 
            'conscious_to_core': self._format_conscious_to_core,
            'conscious_to_subconscious': self._format_conscious_to_subconscious
        }
        
        format_key = f"{self.cluster_location}_to_{target_cluster}"
        formatted_message = formatting_rules[format_key](message)
        
        return await self._deliver_message(target_cluster, formatted_message)
    
    def _format_core_to_subconscious(self, message: Dict) -> Dict:
        """Core speaks to subconscious: Directives and patterns"""
        return {
            "type": "directive",
            "content": message,
            "priority": "background",
            "expectation": "pattern_recognition",
            "timeframe": "flexible"
        }
    
    def _format_subconscious_to_conscious(self, message: Dict) -> Dict:
        """Subconscious speaks to conscious: Suggestions and intuitions"""
        return {
            "type": "suggestion", 
            "content": message,
            "confidence": "intuitive",
            "reasoning": "implicit",
            "urgency": "moderate"
        }
    
    def _format_conscious_to_core(self, message: Dict) -> Dict:
        """Conscious speaks to core: Reports and requests"""
        return {
            "type": "report",
            "content": message,
            "reasoning": "explicit",
            "evidence": "provided",
            "action_requested": True
        }