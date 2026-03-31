"""
Tone Agent — emits regulated emotional signals through the mesh.

Uses an EmotionIntensityRegulator to clamp intensity and logs clips
through the guardian when the requested intensity exceeds the regulated value.
"""

from typing import Any, Dict


class EmotionIntensityRegulator:
    """Regulates emotional intensity to prevent runaway tone signals."""

    def __init__(self, mythrunner: Any) -> None:
        self.mythrunner = mythrunner
        self.max_intensity = 8  # default ceiling

    def regulate(self, intensity: float) -> float:
        """Clamp intensity to the configured ceiling."""
        return min(intensity, self.max_intensity)


class ToneAgent:
    def __init__(self, agent_id: str, mythrunner: Any, guardian: Any) -> None:
        self.agent_id = agent_id
        self.regulator = EmotionIntensityRegulator(mythrunner)
        self.guardian = guardian

    def emit(self, emotion_type: str, intensity: float) -> Dict[str, Any]:
        regulated_intensity = self.regulator.regulate(intensity)
        if intensity > regulated_intensity:
            self.guardian.log("tone_clip", {
                "agent_id": self.agent_id,
                "emotion": emotion_type,
                "requested": intensity,
                "sent": regulated_intensity,
            })
        return {
            "agent_id": self.agent_id,
            "emotion": emotion_type,
            "intensity": regulated_intensity,
        }
