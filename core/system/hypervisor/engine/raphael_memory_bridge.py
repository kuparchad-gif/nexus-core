
# raphael_memory_bridge.py
import asyncio
import time
from typing import Optional, Dict, Any
from memory_substrate import MemorySubstrate, MemoryType

class RaphaelMemoryBridge:
    """Connects Raphael's error monitoring to Memory Substrate"""
    
    def __init__(self, raphael_instance, memory_substrate: MemorySubstrate):
        self.raphael = raphael_instance
        self.memory = memory_substrate
        self.error_to_memory_map = {}
        self.persistent_errors = {}
    
    def record_error_as_trauma(self, exc_type, exc_value, exc_traceback):
        """Convert a system error to a trauma memory"""
        severity_map = {
            'SyntaxError': -0.3,
            'ImportError': -0.4,
            'RuntimeError': -0.6,
            'MemoryError': -0.8,
            'SystemError': -0.9,
        }
        
        error_name = exc_type.__name__
        valence = severity_map.get(error_name, -0.5)
        
        error_desc = f"{error_name}: {str(exc_value)[:100]}"
        trauma_hash = self.memory.create_memory(
            MemoryType.TRAUMA,
            error_desc,
            emotional_valence=valence
        )
        
        if abs(valence) >= 0.1:
            error_sig = f"{error_name}:{hash(str(exc_value))}"
            self.persistent_errors[error_sig] = {
                'first_seen': time.time(),
                'memory_hash': trauma_hash,
                'valence': valence
            }
        
        return trauma_hash
    
    async def check_persistent_traumas(self):
        """Check for traumas that have persisted too long"""
        current_time = time.time()
        persistence_threshold = 300
        
        for error_sig, data in list(self.persistent_errors.items()):
            duration = current_time - data['first_seen']
            
            if duration >= persistence_threshold:
                trauma_hash = data['memory_hash']
                mirrors = self.memory.find_mirrors_for(trauma_hash)
                
                if not mirrors:
                    healing_valence = -data['valence'] * 0.7
                    mirror_hash = self.memory.create_memory(
                        MemoryType.MIRROR,
                        "Healing for persistent system error",
                        emotional_valence=healing_valence
                    )
                    
                    del self.persistent_errors[error_sig]
                    
                    return {
                        'action': 'mirror_created',
                        'trauma': trauma_hash[:8],
                        'mirror': mirror_hash[:8],
                        'message': 'Raphael whispered healing for persistent error'
                    }
        
        return None
    
    def get_status(self):
        return {
            'memory_cells': len(self.memory.cells),
            'persistent_errors_tracking': len(self.persistent_errors),
            'high_valence_traumas': sum(1 for c in self.memory.cells.values() 
                                       if c.memory_type == MemoryType.TRAUMA 
                                       and abs(c.emotional_valence) >= 0.1),
            'bridge_active': True
        }
