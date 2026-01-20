# Systems/VaultCarrier/payload_manager.py

import os
import json

class PayloadManager:
    @staticmethod
    def prepare_payload(paths):
        """Package specified files/folders into memory_payload dict."""
        payload = {}
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r', encoding="utf-8") as f:
                    payload[path] = f.read()
        return payload
