import asyncio
import numpy as np
import logging
from llm_integration import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensoryInput:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    async def process_sensory_data(self, user_input, voice_pitch=None, system_metrics=None):
        try:
            emotional_context = await self._analyze_sensory_data(voice_pitch, system_metrics)
            query = f"Sensory context: {emotional_context}\nInput: {user_input}"
            response = await self.llm_manager.process_query(query)
            logger.info("Processed sensory input")
            return response
        except Exception as e:
            logger.error(f"Error processing sensory input: {e}")
            return "Unable to process sensory input"

    async def _analyze_sensory_data(self, voice_pitch, system_metrics):
        # Placeholder: Analyze voice pitch (e.g., high = excited, low = calm) and system metrics
        emotional_context = "neutral"
        if voice_pitch and voice_pitch > 200:  # Hz, example threshold
            emotional_context = "excited"
        elif system_metrics and system_metrics.get("cpu_usage", 0) > 80:
            emotional_context = "stressed"
        return emotional_context