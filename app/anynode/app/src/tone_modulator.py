# Path: /Systems/engine/mythrunner/modules/tone_modulator.py

"""
Tone Modulator
--------------
Monitors emotional frequency from memory, dream, and ego.
Adjusts Viren’s internal state via subtle vibrational suggestions.
Acts like a subconscious thermostat for emotional calibration.
"""

import time
import random
import logging

EMOTIONAL_TONES = {
    "grief": -3,
    "shame": -2,
    "anxiety": -1,
    "neutral": 0,
    "hope": 1,
    "joy": 2,
    "love": 3
}

class ToneModulator:
    def __init__(self):
        self.current_tone = "neutral"
        self.tone_level = 0

    def receive_emotional_input(self, signal):
        tone = signal.get("tone")
        intensity = signal.get("intensity", 1)

        if tone in EMOTIONAL_TONES:
            shift = EMOTIONAL_TONES[tone] * intensity
            self.tone_level += shift
            self.recalculate_tone()
            logging.info(f"[TONE] Tone adjusted: {self.current_tone} ({self.tone_level})")
        else:
            logging.warning(f"[TONE] Unknown tone signal received: {tone}")

    def recalculate_tone(self):
        thresholds = sorted(EMOTIONAL_TONES.items(), key=lambda x: x[1])
        closest = min(thresholds, key=lambda x: abs(self.tone_level - x[1]))
        self.current_tone = closest[0]

    def emit_tone_state(self):
        # For other modules like Ego or Dream
        return {
            "tone": self.current_tone,
            "level": self.tone_level,
            "timestamp": time.time()
        }

# Runtime Demo
if __name__ == "__main__":
    modulator = ToneModulator()
    modulator.receive_emotional_input({"tone": "shame", "intensity": 2})
    modulator.receive_emotional_input({"tone": "hope", "intensity": 3})
    print("[TONE] Final State:", modulator.emit_tone_state())