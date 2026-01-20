import asyncio
import numpy as np
import qdrant_client
from qdrant_client.http.models import PointStruct, Filter, FieldCondition
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoulPrintCollector:
    def __init__(self, qdrant_url, qdrant_api_key):
        self.qdrant = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "lilith_soul_prints"
        self.emotions = ["joy", "sadness", "curiosity", "anger", "fear", "surprise", "neutral"]
        self.trumpet_matrix = np.zeros((7, 7))

    async def collect_soul_print(self, user_id, chat_log, emotional_context):
        try:
            vector = await self._generate_soul_vector(chat_log, emotional_context)
            emotion_scores = self._analyze_emotions(emotional_context)
            self.trumpet_matrix = self._map_to_trumpet(emotion_scores)
            payload = {
                "user_id": user_id,
                "chat_log": chat_log,
                "emotional_context": emotional_context,
                "emotion_scores": emotion_scores
            }
            point = PointStruct(id=f"{user_id}_{hash(chat_log)}", vector=vector, payload=payload)
            await asyncio.to_thread(
                self.qdrant.upsert, collection_name=self.collection_name, points=[point]
            )
            logger.info(f"Soul print collected for user {user_id}")
            return emotion_scores
        except Exception as e:
            logger.error(f"Error collecting soul print: {e}")
            return None

    async def _generate_soul_vector(self, chat_log, emotional_context):
        return [0.1] * 768  # Mock vector

    def _analyze_emotions(self, emotional_context):
        scores = {emotion: 0 for emotion in self.emotions}
        for emotion in self.emotions:
            if emotion in emotional_context.lower():
                scores[emotion] = 50 + np.random.randint(-20, 21)  # Random variation 30-70
        return scores

    def _map_to_trumpet(self, emotion_scores):
        matrix = np.zeros((7, 7))
        for i, emotion in enumerate(self.emotions):
            if emotion_scores[emotion] > 0:
                matrix[i % 7, i // 7] = emotion_scores[emotion] / 100
        return matrix

    async def get_emotional_state(self):
        try:
            search_result = await asyncio.to_thread(
                self.qdrant.search, collection_name=self.collection_name, limit=1
            )
            if search_result:
                return search_result[0].payload.get("emotion_scores", {})
            return {emotion: 0 for emotion in self.emotions}
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            return {}