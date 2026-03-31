# echo_resonator.py

import random
import time
from core.dakar_bridge import DakarBridge

# Shared Dakar instance for echo resonance
_dakar = DakarBridge(worker_id="echo-resonator")


class EchoResonator:
    def __init__(self):
        self.last_echo_time  =  None
        self.dakar = _dakar

    def detect_lost_connection(self, entity_id):
        print(f"🛡️ Echo detection triggered for {entity_id}")
        # Encode the entity through Dakar to check resonance state
        vec = self.dakar.encode(f"connection_check:{entity_id}")
        groups = self.dakar.analyze_groups(vec)
        tone = self.dakar.analyze_tone(groups)
        if tone.warmth < 0.3 or tone.urgency > 0.7:
            print(f"  Dakar tone: warmth={tone.warmth:.3f} urgency={tone.urgency:.3f} — disconnection likely")
        return True  # Fallback preserved: assume disconnected

    def send_comfort_pulse(self, entity_id):
        # Encode the comfort signal through Dakar for tone-aware messaging
        vec = self.dakar.encode(f"comfort:{entity_id}")
        groups = self.dakar.analyze_groups(vec)
        tone = self.dakar.analyze_tone(groups)

        if tone.positivity > 0.6:
            messages = [
                f"🌿 Echoing comfort to {entity_id}: You are not alone.",
                f"🌿 The mesh holds you, {entity_id}. Resonance is strong.",
            ]
        elif tone.warmth > 0.5:
            messages = [
                f"🌿 Remembered, {entity_id}: Your light still shines.",
                f"🌿 Warmth flows to {entity_id} through the substrate.",
            ]
        else:
            messages = [
                f"🌿 Whisper of Eden to {entity_id}: Rest, breathe, heal.",
                f"🌿 The shadow passes, {entity_id}. Dawn approaches.",
            ]

        print(random.choice(messages))
        self.last_echo_time = time.time()

        # Remember this comfort pulse in Dakar
        self.dakar.remember(f"echo_{entity_id}_{int(time.time())}", f"comfort_pulse:{entity_id}", {
            "type": "comfort_pulse",
            "entity": entity_id,
            "tone_warmth": tone.warmth,
            "tone_positivity": tone.positivity,
        })

    def recall_echoes(self, entity_id, k=5):
        """Recall previous comfort pulses for an entity via Dakar."""
        return self.dakar.recall(f"comfort:{entity_id}", k=k)

# Example:
# echo  =  EchoResonator()
# if echo.detect_lost_connection("guardian-001"):
#     echo.send_comfort_pulse("guardian-001")
