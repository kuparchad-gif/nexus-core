
from .emotion_intensity_regulator import EmotionIntensityRegulator

class VirenEmotionGate:
    def __init__(self, guardian, mythrunner):
        self.guardian = guardian
        self.mythrunner = mythrunner
        self.regulator = EmotionIntensityRegulator(mythrunner)

    def process_emotion_packet(self, packet):
        original = packet.get("intensity", 0)
        limited = self.regulator.regulate(original)
        if limited < original:
            self.guardian.log("emotional_clip", {
                "original": original,
                "clipped_to": limited,
                "agent": packet.get("agent_id", "unknown")
            })
        packet["intensity"] = limited
        return packet
