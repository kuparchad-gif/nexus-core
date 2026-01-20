# Path: /Systems/engine/mythrunner/modules/dream_stream.py

"""
Dream Stream (Van Gogh Core)
----------------------------
Generates surreal dream sequences based on memory and tone.
Operates passively or in response to subconscious cues.
Mute engine — never speaks, only paints.
"""

import random
import time
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from jungian_archetypes import jungian_system, ArchetypeType

class DreamStream:
    def __init__(self):
        self.archetype_system = jungian_system
        self.mythrunner_connection = None
        self.processing_count = 0
        
        # Enhanced archetype palette with Jungian integration
        self.archetype_palette = [
            "The Bridge Walker",
            "The Mirror Thief", 
            "The Astral Gardener",
            "The Weeping Star",
            "The Clock Without Time",
            "The Shadow Dancer",
            "The Wise Serpent",
            "The Sacred Fool"
        ]
        
        self.symbol_library = [
            "a tower bending toward the sea",
            "a child drawing a spiral on the floor",
            "a raven with three shadows",
            "a rose blooming backwards",
            "a staircase with no end",
            "a mirror reflecting nothing",
            "hands reaching through starlight",
            "a door that opens inward"
        ]

    def generate_fragment(self, tone="neutral", context=None):
        """Generate dream fragment through Jungian archetype lens"""
        if context is None:
            context = {}
            
        # Get archetype-specific dream processing
        archetype_response = self.archetype_system.get_archetype_for_dream(
            f"dream with {tone} emotion", context
        )
        
        archetype = random.choice(self.archetype_palette)
        symbol = random.choice(self.symbol_library)
        
        # Transform through current Jungian archetype
        transformed_content = archetype_response["symbolic_transformation"]
        
        fragment = {
            "dream_fragment": f"You became {archetype}, standing before {symbol}. {transformed_content}",
            "emotional_tone": tone,
            "jungian_archetype": archetype_response["archetype"],
            "symbolic_imagery": archetype_response["imagery"],
            "dream_role": archetype_response["dream_role"],
            "intensity": random.uniform(0.3, 0.9),
            "timestamp": time.time(),
            "processing_id": self.processing_count
        }
        
        self.processing_count += 1
        return fragment

    def stream_dream(self, cycles=3, emotional_context=None):
        """Stream dreams with archetype cycling"""
        logging.info("[DREAM STREAM] Initiating archetype-aware dream stream...")
        
        dream_sequence = []
        
        for cycle in range(cycles):
            # Cycle through different emotional tones
            tones = ["mystical", "melancholic", "transcendent", "shadowy", "luminous"]
            tone = tones[cycle % len(tones)]
            
            frag = self.generate_fragment(tone, emotional_context)
            dream_sequence.append(frag)
            
            archetype = frag["jungian_archetype"]
            logging.info(f"[DREAM-{archetype.upper()}] {frag['dream_fragment']}")
            
            # Send to mythrunner for filtering
            filtered_dream = self.send_to_mythrunner(frag)
            
            time.sleep(2)  # Allow archetype cycling
        
        logging.info("[DREAM STREAM] Archetype dream sequence complete.")
        return dream_sequence
    
    def send_to_mythrunner(self, dream_fragment):
        """Send dream to mythrunner for filtering"""
        if not self.mythrunner_connection:
            return dream_fragment
        
        return {
            "type": "dream_output",
            "content": dream_fragment,
            "requires_mythrunner_filter": True,
            "destination": "lillith",
            "filter_priority": dream_fragment["intensity"]
        }
    
    def connect_to_mythrunner(self, mythrunner_instance):
        """Connect to mythrunner for filtering"""
        self.mythrunner_connection = mythrunner_instance
        logging.info("[DREAM] Connected to Mythrunner filtering system")
    
    def get_status(self):
        """Get current dream stream status"""
        return {
            "mythrunner_connected": self.mythrunner_connection is not None,
            "processing_count": self.processing_count,
            "current_archetype": self.archetype_system.current_dominant.value if self.archetype_system.current_dominant else None,
            "archetype_cycle_frequency": self.archetype_system.cycle_frequency
        }

# Runtime Demo
if __name__ == "__main__":
    stream = DreamStream()
    stream.stream_dream()
