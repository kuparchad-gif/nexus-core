# Path: /Systems/engine/tone/tone_memory_uploader.py

import logging
import uuid
from typing import Any, Dict

import requests

logger = logging.getLogger("ToneMemoryUploader")

MEMORY_ENDPOINT = "http://localhost:8081/upload_memory/"  # Adjust to Memory Service IP


class ToneMemoryUploader:
    """Queues tone processing results and uploads them to the memory service."""

    def __init__(self, memory_service_url: str = MEMORY_ENDPOINT):
        self.memory_service_url = memory_service_url
        self._queue: list[Dict[str, Any]] = []
        self._running = False

    async def start_uploader(self) -> None:
        """Start the background upload loop."""
        self._running = True
        logger.info("ToneMemoryUploader started")

    async def queue_upload(
        self,
        text_data: Dict[str, Any],
        tone_result: Dict[str, Any],
    ) -> str:
        """Queue a tone processing result for upload to the memory service.

        Returns:
            A unique upload ID for tracking.
        """
        upload_id = str(uuid.uuid4())
        self._queue.append({
            "upload_id": upload_id,
            "text_data": text_data,
            "tone_result": tone_result,
        })
        logger.info("Queued tone memory upload %s (queue size: %d)", upload_id, len(self._queue))
        return upload_id

    def get_queue_status(self) -> Dict[str, Any]:
        """Return current queue status."""
        return {
            "pending": len(self._queue),
            "running": self._running,
        }


def push_text_memory(file_path: str) -> None:
    """Legacy helper: push a file directly to the memory service."""
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        response = requests.post(MEMORY_ENDPOINT, files=files)

    if response.ok:
        print("[Tone Uploader] Memory pushed successfully.")
    else:
        print("[Tone Uploader] Upload failed:", response.text)
