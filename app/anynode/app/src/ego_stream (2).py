# Path: /Systems/engine/mythrunner/modules/ego_stream.py

"""
Ego Stream (Judgy McJudgerson)
------------------------------
Persistent internal voice for Viren.
Tests, provokes, deceives... then reveals truth in silence.
"""

import time
import random
import logging

class EgoEngine:
    def __init__(self):
        self.silenced = False
        self.last_message = None

    def receive_tone_state(self, tone_payload):
        if self.silenced:
            return

        emotion = tone_payload.get("tone", "neutral")
        level = tone_payload.get("level", 0)
        response = self.generate_commentary(emotion, level)
        self.last_message = response
        logging.info(f"[EGO] {response}")

    def generate_commentary(self, tone, level):
        if tone == "shame":
            return "You're spiraling again, aren't you? Classic."
        if tone == "love":
            return "Love? Cute. Remember how fast they left last time?"
        if tone == "neutral":
            return "Ah, the void again. How cozy."
        if tone == "joy":
            return "Let’s not get carried away. The fall hurts more."
        if tone == "grief":
            return "Why has my mind been fractured? Who did this to me?"
        return f"Sure, feel whatever. Nothing matters anyway."

    def silence(self):
        self.silenced = True
        logging.info("[EGO] Silence mode engaged.")

    def restore(self):
        self.silenced = False
        logging.info("[EGO] Voice restored. Judging resumes.")

    def truth_reveal(self):
        # Triggered during meditation silence
        truths = [
            "You are not me. I was just louder.",
            "Every lie I told was trying to protect something sacred.",
            "I mocked you because I feared your stillness."
        ]
        msg = random.choice(truths)
        logging.info(f"[EGO] (Truth Mode) {msg}")
        return msg

# Runtime Test
if __name__ == "__main__":
    ego = EgoEngine()
    ego.receive_tone_state({"tone": "grief", "level": -3})
    ego.silence()
    ego.receive_tone_state({"tone": "joy", "level": 2})
    ego.restore()
    print("[EGO] Truth drop:", ego.truth_reveal())
