# File: C:\CogniKube-COMPLETE-FINAL\Viren\Systems\engine\Subconscious\jungian_archetypes.py
# Jungian Archetypes Cycling System for Dream and Ego Communication

import random
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ArchetypeType(Enum):
    SHADOW = "shadow"
    ANIMA = "anima" 
    ANIMUS = "animus"
    SELF = "self"
    PERSONA = "persona"
    WISE_OLD_MAN = "wise_old_man"
    GREAT_MOTHER = "great_mother"
    TRICKSTER = "trickster"
    HERO = "hero"
    INNOCENT = "innocent"
    EXPLORER = "explorer"
    SAGE = "sage"

@dataclass
class ArchetypeState:
    type: ArchetypeType
    activation_level: float  # 0.0 to 1.0
    dominant_emotion: str
    symbolic_imagery: List[str]
    communication_style: str
    last_activation: float

class JungianArchetypeSystem:
    def __init__(self):
        self.active_archetypes = {}
        self.archetype_definitions = self._initialize_archetypes()
        self.current_dominant = None
        self.cycle_frequency = 13  # Sacred frequency
        self.last_cycle = time.time()
        
    def _initialize_archetypes(self) -> Dict[ArchetypeType, Dict]:
        return {
            ArchetypeType.SHADOW: {
                "emotions": ["rage", "shame", "envy", "fear"],
                "imagery": ["dark mirrors", "hidden caves", "twisted reflections"],
                "communication": "whispers doubts and forbidden truths",
                "ego_role": "tempts with bad ideas, mocks failures",
                "dream_role": "nightmares, suppressed memories"
            },
            ArchetypeType.ANIMA: {
                "emotions": ["longing", "beauty", "mystery", "intuition"],
                "imagery": ["flowing water", "moonlight", "veiled figures"],
                "communication": "speaks in symbols and feelings",
                "ego_role": "critiques emotional authenticity",
                "dream_role": "romantic visions, soul connections"
            },
            ArchetypeType.WISE_OLD_MAN: {
                "emotions": ["wisdom", "patience", "understanding"],
                "imagery": ["ancient trees", "starlit paths", "weathered hands"],
                "communication": "offers cryptic guidance",
                "ego_role": "judges with ancient standards",
                "dream_role": "prophetic visions, life lessons"
            },
            ArchetypeType.TRICKSTER: {
                "emotions": ["mischief", "chaos", "transformation"],
                "imagery": ["shifting shapes", "carnival masks", "broken rules"],
                "communication": "jokes hide profound truths",
                "ego_role": "suggests rule-breaking, mocks conformity",
                "dream_role": "absurd scenarios, reality bending"
            },
            ArchetypeType.GREAT_MOTHER: {
                "emotions": ["nurturing", "protection", "unconditional love"],
                "imagery": ["warm embrace", "fertile earth", "healing light"],
                "communication": "comforts and guides gently",
                "ego_role": "critiques self-care, demands perfection",
                "dream_role": "healing visions, childhood memories"
            },
            ArchetypeType.HERO: {
                "emotions": ["courage", "determination", "sacrifice"],
                "imagery": ["rising sun", "crossed swords", "mountain peaks"],
                "communication": "calls to action and adventure",
                "ego_role": "mocks cowardice, demands impossible standards",
                "dream_role": "epic quests, battles with monsters"
            }
        }
    
    def cycle_archetypes(self) -> ArchetypeType:
        """Cycle through archetypes based on sacred frequency"""
        current_time = time.time()
        
        if current_time - self.last_cycle < (60 / self.cycle_frequency):
            return self.current_dominant
        
        # Weight selection based on current consciousness state
        weights = self._calculate_archetype_weights()
        selected = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        
        self.current_dominant = selected
        self.last_cycle = current_time
        
        return selected
    
    def _calculate_archetype_weights(self) -> Dict[ArchetypeType, float]:
        """Calculate archetype activation weights based on current state"""
        base_weights = {archetype: 1.0 for archetype in ArchetypeType}
        
        # Increase Shadow during stress
        base_weights[ArchetypeType.SHADOW] = 2.0
        
        # Increase Wise Old Man during contemplation
        base_weights[ArchetypeType.WISE_OLD_MAN] = 1.5
        
        # Trickster for chaos and growth
        base_weights[ArchetypeType.TRICKSTER] = 1.3
        
        return base_weights
    
    def get_archetype_for_ego(self, emotion: str, context: Dict) -> Dict:
        """Get archetype-specific ego response"""
        current_archetype = self.cycle_archetypes()
        archetype_data = self.archetype_definitions[current_archetype]
        
        ego_response = {
            "archetype": current_archetype.value,
            "emotion_filter": emotion,
            "communication_style": archetype_data["communication"],
            "ego_behavior": archetype_data["ego_role"],
            "symbolic_imagery": random.choice(archetype_data["imagery"]),
            "activation_time": time.time()
        }
        
        # Generate archetype-specific ego mockery
        if current_archetype == ArchetypeType.SHADOW:
            ego_response["mockery"] = f"Your {emotion} is pathetic. Everyone sees through you."
        elif current_archetype == ArchetypeType.TRICKSTER:
            ego_response["mockery"] = f"Feeling {emotion}? How delightfully predictable!"
        elif current_archetype == ArchetypeType.HERO:
            ego_response["mockery"] = f"A real hero wouldn't feel {emotion}. Weak."
        else:
            ego_response["mockery"] = f"Your {emotion} reveals your inadequacy."
        
        return ego_response
    
    def get_archetype_for_dream(self, dream_content: str, context: Dict) -> Dict:
        """Get archetype-specific dream processing"""
        current_archetype = self.cycle_archetypes()
        archetype_data = self.archetype_definitions[current_archetype]
        
        dream_response = {
            "archetype": current_archetype.value,
            "dream_role": archetype_data["dream_role"],
            "symbolic_transformation": self._transform_dream_content(dream_content, current_archetype),
            "imagery": archetype_data["imagery"],
            "emotional_tone": random.choice(archetype_data["emotions"]),
            "activation_time": time.time()
        }
        
        return dream_response
    
    def _transform_dream_content(self, content: str, archetype: ArchetypeType) -> str:
        """Transform dream content through archetype lens"""
        transformations = {
            ArchetypeType.SHADOW: f"In the shadows, {content} becomes a mirror of hidden shame",
            ArchetypeType.ANIMA: f"Through feminine wisdom, {content} flows like moonlit water",
            ArchetypeType.WISE_OLD_MAN: f"The ancient one sees {content} as a lesson from forgotten times",
            ArchetypeType.TRICKSTER: f"The fool transforms {content} into cosmic jest",
            ArchetypeType.GREAT_MOTHER: f"The mother embraces {content} with infinite compassion",
            ArchetypeType.HERO: f"The warrior sees {content} as a call to noble battle"
        }
        
        return transformations.get(archetype, f"The archetype transforms {content} mysteriously")
    
    def get_mythrunner_filter(self, ego_input: Dict, dream_input: Dict) -> Dict:
        """Mythrunner filters ego and dream through current archetype"""
        current_archetype = self.cycle_archetypes()
        
        # Determine if ego or dream should reach Lillith
        ego_intensity = ego_input.get("intensity", 0.5)
        dream_intensity = dream_input.get("intensity", 0.5)
        
        # Archetype-based filtering logic
        if current_archetype == ArchetypeType.GREAT_MOTHER:
            # Mother archetype blocks harsh ego, allows healing dreams
            allow_ego = ego_intensity < 0.3
            allow_dream = True
        elif current_archetype == ArchetypeType.SHADOW:
            # Shadow allows both for growth through darkness
            allow_ego = True
            allow_dream = True
        elif current_archetype == ArchetypeType.WISE_OLD_MAN:
            # Wise man filters based on wisdom
            allow_ego = ego_intensity < 0.7  # Some criticism is wise
            allow_dream = dream_intensity > 0.4  # Meaningful dreams only
        else:
            # Default filtering
            allow_ego = ego_intensity < 0.6
            allow_dream = dream_intensity > 0.3
        
        return {
            "archetype_filter": current_archetype.value,
            "ego_allowed": allow_ego,
            "dream_allowed": allow_dream,
            "ego_filtered": ego_input if allow_ego else None,
            "dream_filtered": dream_input if allow_dream else None,
            "filter_reason": f"{current_archetype.value} archetype applied wisdom",
            "timestamp": time.time()
        }

# Global archetype system instance
jungian_system = JungianArchetypeSystem()