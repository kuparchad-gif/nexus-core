# Path: /Systems/engine/text/text_memory_uploader.py

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("TextMemoryUploader")

MEMORY_ENDPOINT = "http://localhost:8081/upload_memory/"  # Adjust to Memory Service IP


class TextMemoryUploader:
    """Queues text processing results and uploads them to the memory service."""

    def __init__(self, memory_service_url: str = MEMORY_ENDPOINT):
        self.memory_service_url = memory_service_url
        self._queue: list[Dict[str, Any]] = []
        self._running = False

    async def start_uploader(self) -> None:
        """Start the background upload loop."""
        self._running = True
        logger.info("TextMemoryUploader started")

    async def queue_upload(
        self,
        text_data: Dict[str, Any],
        processing_result: Dict[str, Any],
    ) -> str:
        """Queue a text processing result for upload to the memory service.

        Returns:
            A unique upload ID for tracking.
        """
        upload_id = str(uuid.uuid4())
        self._queue.append({
            "upload_id": upload_id,
            "text_data": text_data,
            "processing_result": processing_result,
        })
        logger.info("Queued text memory upload %s (queue size: %d)", upload_id, len(self._queue))
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
        print("[Text Uploader] Memory pushed successfully.")
    else:
        print("[Text Uploader] Upload failed:", response.text)
