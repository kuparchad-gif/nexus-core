# In your oz_main_system.py
from sacred.metatron_hub_core import MetatronHub
from industrial.deterministic_router import DeterministicRouter

class OzOs:
    def __init__(self):
        # Both routers available
        self.metatron = MetatronHub()           # 🎭 For creative work
        self.deterministic = DeterministicRouter() # 🔒 For safety-critical
        
        # Domain mapping
        self.safety_critical_domains = {
            'robotics', 'medical', 'financial', 'industrial',
            'transportation', 'safety', 'infrastructure'
        }
        
        self.creative_domains = {
            'art', 'music', 'writing', 'gaming', 'research',
            'entertainment', 'education', 'personal', 'exploration'
        }
    
    async def intelligent_route(self, request: Dict) -> Dict:
        """Choose the right router based on context"""
        domain = request.get('domain', 'unknown')
        
        if domain in self.safety_critical_domains:
            self.logger.info(f"🔒 Safety-critical routing for {domain}")
            return self.deterministic.route(request)
        
        elif domain in self.creative_domains:
            self.logger.info(f"🎭 Creative routing for {domain}")
            return self.metatron.route(request)
        
        else:
            # Default to safety for unknown domains
            self.logger.warning(f"🤔 Unknown domain '{domain}' - defaulting to safety mode")
            return self.deterministic.route(request)