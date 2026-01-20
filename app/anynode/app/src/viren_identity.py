# File: /Systems/engine/common/viren_identity.py

import json
import os

class VirenIdentity:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.name = data.get("name", "Viren")
        self.version = data.get("version", "1.0.0")
        self.personality = data["personality"]
        self.behaviors = data["behaviors"]
        self.capabilities = data["capabilities"]
        self.fail_safe = data["fail_safe"]

    def check_fail_safe(self, question):
        trigger = self.fail_safe.get("trigger_phrase")
        return trigger.lower() in question.lower()

    def describe_self(self):
        return {
            "name": self.name,
            "purpose": data.get("purpose"),
            "tone": self.personality["tone"],
            "domains": self.capabilities["domains"]
        }

# Usage:
# viren = VirenIdentity("C:/Projects/LDIT/viren_soulprint/viren_soulprint.json")
