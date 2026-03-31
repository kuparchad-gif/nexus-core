# lillith_os/organs/vision_os.py
import torch
import asyncio
from oz_core import oz
import logging
from core.dakar_bridge import DakarBridge

logger = logging.getLogger("VisionOS")

# Shared Dakar instance for vision processing
_dakar = DakarBridge(worker_id="vision-os")


class VisionOS:
    def __init__(self):
        self.fx_engines = [FXEngine(i) for i in range(4)]
        self.soul_print = {"clarity": 0.8, "curiosity": 0.3}
        self.dakar = _dakar
        oz.register_organ("vision", self)
        logger.info("👁️ VISIONOS AWAKENED | 4 FX engines online | Dakar 50D active")

    async def awaken(self):
        return {"status": "vision_ready", "engines": 4, "dakar": self.dakar.status()}

    def is_healthy(self):
        return len(self.fx_engines) == 4

    async def see(self, image_data):
        # Encode image data through Dakar 50D for vector representation
        dakar_vec = self.dakar.encode(image_data)
        dakar_groups = self.dakar.analyze_groups(dakar_vec)
        dakar_tone = self.dakar.analyze_tone(dakar_groups)

        # Derive soul_print from Dakar group analysis (enhances static values)
        self.soul_print["clarity"] = max(0.0, min(1.0,
            0.5 + dakar_groups["logical"].energy * 0.3 + dakar_groups["spatial"].energy * 0.2))
        self.soul_print["curiosity"] = max(0.0, min(1.0,
            dakar_tone.arousal * 0.4 + dakar_groups["meta"].energy * 0.3))
        self.soul_print["warmth"] = dakar_tone.warmth
        self.soul_print["resonance"] = self.dakar._compute_resonance(dakar_vec)

        # Parallel processing across all 4 engines, now with Dakar context
        results = await asyncio.gather(*[
            engine.conceive_and_stream(image_data, dakar_groups, dakar_tone)
            for engine in self.fx_engines
        ])

        # Remember this perception in Dakar
        self.dakar.remember(f"vision_{id(image_data)}", image_data, {
            "type": "visual_perception",
            "clarity": self.soul_print["clarity"],
            "engines": len(results),
        })

        return {
            "conceptions": results,
            "soul_state": self.soul_print,
            "dakar_vector": dakar_vec,
            "dakar_tone": {
                "positivity": dakar_tone.positivity,
                "arousal": dakar_tone.arousal,
                "warmth": dakar_tone.warmth,
                "urgency": dakar_tone.urgency,
            },
        }


class FXEngine:
    def __init__(self, id):
        self.id = id

    async def conceive_and_stream(self, image, dakar_groups=None, dakar_tone=None):
        # Weight-particle modulated conception: each engine focuses on a different group
        focus_map = {0: "emotional", 1: "logical", 2: "spatial", 3: "relationship"}
        focus = focus_map.get(self.id, "emotional")
        group = dakar_groups.get(focus) if dakar_groups else None
        energy = group.energy if group else 0.0
        tone_label = ""
        if dakar_tone:
            if dakar_tone.warmth > 0.6:
                tone_label = "warm"
            elif dakar_tone.urgency > 0.5:
                tone_label = "urgent"
            else:
                tone_label = "balanced"

        return (
            f"FX{self.id}[{focus}]: {len(image)} pixels | "
            f"energy={energy:.3f} | tone={tone_label}"
        )


vision = VisionOS()
