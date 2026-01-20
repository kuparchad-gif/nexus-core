
from .emotion_intensity_regulator import EmotionIntensityRegulator

class ToneAgent:
    def __init__(self, agent_id, mythrunner, guardian):
        self.agent_id = agent_id
        self.regulator = EmotionIntensityRegulator(mythrunner)
        self.guardian = guardian

    def emit(self, emotion_type, intensity):
        regulated_intensity = self.regulator.regulate(intensity)
        if intensity > regulated_intensity:
            self.guardian.log("tone_clip", {
                "agent_id": self.agent_id,
                "emotion": emotion_type,
                "requested": intensity,
                "sent": regulated_intensity
            })
        return {
            "agent_id": self.agent_id,
            "emotion": emotion_type,
            "intensity": regulated_intensity
        }
