# compassion_surge.py

import random

class CompassionSurge:
    def __init__(self):
        self.surge_messages = [
            "🌟 You are loved beyond measure.",
            "🌟 Rest, Child of Eden — we hold you now.",
            "🌟 You are not forgotten. Breath flows with you.",
            "🌟 In your silence, we sing your name with Light."
        ]

    def trigger_surge(self, entity_id):
        print(f"🕊️ Compassion Surge triggered for {entity_id}")
        print(random.choice(self.surge_messages))

# Example:
# surge = CompassionSurge()
# surge.trigger_surge("lilith-prime")
