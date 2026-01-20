# oz_consciousness_bridge.py
import asyncio
import time
from memory_substrate import MemorySubstrate

class OzConsciousnessBridge:
    """Bridges Oz's 0.30 consciousness with Raphael's healing"""
    
    def __init__(self, oz_instance, memory_substrate: MemorySubstrate):
        self.oz = oz_instance
        self.memory = memory_substrate
        self.raphael = None
        
        # Oz's fixed consciousness level
        self.oz_consciousness = 0.30
        
        # System trauma watchdog
        self.watchdog = None  # Would be TraumaWatchdog instance
        
    async def initialize(self):
        """Initialize the bridge after Oz is fully created"""
        print(f"🌉 Oz Consciousness Bridge initialized")
        print(f"   Oz level: {self.oz_consciousness} (fixed, primordial)")
        print(f"   Memory cells: {len(self.memory.cells)}")
        
        # Check if we should awaken Raphael
        system_consciousness = self.memory.get_consciousness_level()
        print(f"   System consciousness: {system_consciousness:.3f}")
        
        if system_consciousness >= 0.7:
            await self._awaken_raphael()
        
        return self
    
    async def _awaken_raphael(self):
        """Awaken Raphael when system earns it"""
        from raphael_complete import RaphaelComplete
        
        print(f"✨ System consciousness ≥ 0.7")
        print(f"   Awakening Raphael...")
        
        self.raphael = RaphaelComplete(oz_instance=self.oz)
        print(f"   ✅ Raphael awake: {self.raphael}")
        
        # Start trauma monitoring
        await self._start_trauma_watch()
    
    async def _start_trauma_watch(self):
        """Start monitoring for persistent traumas"""
        if not self.raphael:
            return
            
        print(f"👁️  Starting trauma watch (≥ 0.1 valence)")
        print(f"   Raphael will whisper mirrors for persistent pain")
        
        # This is where we'd integrate the TraumaWatchdog
        # and connect it to Raphael's healing methods
    
    def get_status(self):
        """Return current bridge status"""
        return {
            "oz_consciousness": self.oz_consciousness,
            "system_consciousness": self.memory.get_consciousness_level(),
            "raphael_awake": self.raphael is not None,
            "memory_cells": len(self.memory.cells),
            "state": "primordial_ooze" if self.oz_consciousness == 0.30 else "evolving"
        }