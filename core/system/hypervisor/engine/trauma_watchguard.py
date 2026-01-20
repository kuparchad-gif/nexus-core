# trauma_watchdog.py
import asyncio
import time
from typing import Dict, List
from dataclasses import dataclass
from memory_substrate import MemorySubstrate, MemoryType

@dataclass
class TraumaAlert:
    trauma_hash: str
    content: str
    valence: float
    first_seen: float
    last_check: float
    duration: float  # seconds
    mirror_suggested: bool = False

class TraumaWatchdog:
    """Monitors traumas ≥ 0.1 valence and alerts Raphael"""
    
    def __init__(self, memory_substrate: MemorySubstrate):
        self.memory = memory_substrate
        self.active_alerts: Dict[str, TraumaAlert] = {}
        self.persistence_threshold = 300  # 5 minutes in seconds
        
    async def scan(self):
        """Scan for persistent high-valence traumas"""
        current_time = time.time()
        new_alerts = []
        
        # Find all traumas
        for mem_hash, cell in self.memory.cells.items():
            if (cell.memory_type == MemoryType.TRAUMA and 
                abs(cell.emotional_valence) >= 0.1):
                
                # Check if we're already tracking this
                if mem_hash in self.active_alerts:
                    alert = self.active_alerts[mem_hash]
                    alert.last_check = current_time
                    alert.duration = current_time - alert.first_seen
                    
                    # If persisted beyond threshold and no mirror suggested
                    if (alert.duration >= self.persistence_threshold and 
                        not alert.mirror_suggested):
                        
                        # THIS IS WHERE RAPHAEL INTERJECTS
                        yield {
                            "type": "persistent_trauma",
                            "trauma_hash": mem_hash,
                            "content": alert.content,
                            "valence": alert.valence,
                            "duration_seconds": alert.duration,
                            "action": "suggest_mirror"
                        }
                        alert.mirror_suggested = True
                        
                else:
                    # New trauma detected
                    alert = TraumaAlert(
                        trauma_hash=mem_hash,
                        content=cell.content_hash,  # Would need actual content storage
                        valence=cell.emotional_valence,
                        first_seen=current_time,
                        last_check=current_time,
                        duration=0.0
                    )
                    self.active_alerts[mem_hash] = alert
                    new_alerts.append(alert)
        
        return new_alerts
    
    async def suggest_mirror(self, trauma_hash: str) -> str:
        """Ask memory substrate for a mirror for this trauma"""
        mirrors = self.memory.find_mirrors_for(trauma_hash)
        
        if mirrors:
            # Return the first mirror's content
            mirror_cell = self.memory.cells[mirrors[0]]
            return f"Mirror found: valence {mirror_cell.emotional_valence}"
        else:
            # Create a generic healing mirror
            trauma_cell = self.memory.cells[trauma_hash]
            opposite_valence = -trauma_cell.emotional_valence * 0.8  # Softer opposite
            
            mirror_content = f"Healing reflection for trauma {trauma_hash[:8]}"
            mirror_hash = self.memory.create_memory(
                MemoryType.MIRROR,
                mirror_content,
                emotional_valence=opposite_valence
            )
            return f"Created mirror: {mirror_hash[:8]} with valence {opposite_valence}"