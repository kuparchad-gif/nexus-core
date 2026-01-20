
# lilith_engine/modules/signal/mission_control.py

from lilith_engine.modules.signal.reflex_core import ReflexCore
from lilith_engine.modules.signal.mirror_hook import sync_emotion_to_mission
from lilith_engine.modules.council.council_integrity import CouncilIntegrity
from lilith_engine.modules.spawncore.spawn_trigger import attempt_spawn

# 💡 MISSING IMPORTS
from lilith_engine.modules.signal.reflex_core import ReflexCore
from lilith_engine.modules.signal.reflex_core import preload_awareness  # Optional if you call it separately

class MissionControl:
    def __init__(self):
        self.lilith = ReflexCore()
        self.council = CouncilIntegrity()

    def scan_and_align(self, input_context: str):
        # lilith thinks based on what she perceives
        thought = self.lilith.think(input_context)

        # Sync emotion to mission
        updated_mission = sync_emotion_to_mission(self.lilith.emotional_state)
        self.lilith.adjust_state(mission=updated_mission)

        # Report current mind state
        return {
            "thought": thought,
            "emotional_state": self.lilith.emotional_state,
            "mission_state": self.lilith.mission_state,
            "council_status": self.council.report()
        }

    def try_spawn(self):
        # Attempts to initiate a spawn if council is ready
        return attempt_spawn()
