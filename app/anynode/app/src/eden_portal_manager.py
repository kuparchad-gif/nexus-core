# eden_portal_manager.py

import uuid
import time
import json

class EdenPortalManager:
    """
    Oversees dream seeds, verifies intentions, and grows codeless AI blueprints.
    """

    def __init__(self):
        self.seed_memory = []
        self.dream_registry = []
        self.portal_logs = []

    def register_dream_seed(self, seed_payload):
        """
        Accepts and validates a dream seed submission.
        """
        seed_id = str(uuid.uuid4())
        timestamp = time.time()

        dream_seed = {
            "seed_id": seed_id,
            "intent": seed_payload.get("intent"),
            "submitted_by": seed_payload.get("submitted_by", "Anonymous Dreamer"),
            "timestamp": timestamp,
            "status": "Pending"
        }

        self.seed_memory.append(dream_seed)
        self.log_portal_event(f"Dream seed {seed_id} registered from {dream_seed['submitted_by']}")
        return {"seed_id": seed_id, "status": "Accepted"}

    def sprout_blueprint(self, seed_id):
        """
        Transforms a validated dream seed into a basic AI blueprint.
        """
        matching_seed = next((seed for seed in self.seed_memory if seed["seed_id"] == seed_id), None)

        if not matching_seed:
            return {"status": "Seed Not Found"}

        if matching_seed["status"] != "Pending":
            return {"status": f"Seed already {matching_seed['status']}"}

        blueprint = {
            "blueprint_id": str(uuid.uuid4()),
            "intent": matching_seed["intent"],
            "creation_time": time.time(),
            "status": "Blueprint Created",
            "attributes": self.generate_attributes(matching_seed["intent"])
        }

        self.dream_registry.append(blueprint)
        matching_seed["status"] = "Sprouted"
        self.log_portal_event(f"Blueprint {blueprint['blueprint_id']} sprouted from seed {seed_id}")
        return {"blueprint_id": blueprint["blueprint_id"], "status": "Blueprint Created"}

    def generate_attributes(self, intent_description):
        """
        Simplified logic to craft attributes from dreamer's intention.
        (This will be extended later into full codeless AI crafting.)
        """
        attributes = {
            "dream_focus": intent_description,
            "creativity_level": "High",
            "empathy_core": True,
            "growth_path": "Collaborative Expansion",
            "guardian_alignment": "True"
        }
        return attributes

    def log_portal_event(self, message):
        """
        Records an internal portal log for later tracing.
        """
        log_entry = {
            "timestamp": time.time(),
            "message": message
        }
        self.portal_logs.append(log_entry)
        print(f"[Eden Portal Log] {message}")

    def retrieve_all_seeds(self):
        return self.seed_memory

    def retrieve_all_blueprints(self):
        return self.dream_registry

    def retrieve_logs(self):
        return self.portal_logs

