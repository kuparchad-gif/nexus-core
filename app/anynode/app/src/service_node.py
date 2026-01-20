# service_node.py
import os
import json
import random
import time

class ServiceNode:
    def __init__(self, role_config_path="Config/roles/default_role.json"):
        self.role_config_path = role_config_path
        self.role = None
        self.node_id = f"node-{random.randint(1000,9999)}"
        self.pulse_interval = 13
        self.skills = {}

    def initialize(self):
        self.load_role()
        self.load_skills()

    def load_role(self):
        if not os.path.exists(self.role_config_path):
            print("❗ No role config found. Defaulting to observer.")
            self.role = "observer"
        else:
            with open(self.role_config_path, "r") as f:
                data = json.load(f)
                self.role = data.get("role", "observer")

    def load_skills(self):
        skill_path = "memory/skill_packages/skills.json"
        if os.path.exists(skill_path):
            with open(skill_path, "r") as f:
                self.skills = json.load(f)
            print(f"🧠 Loaded {len(self.skills)} skills.")
        else:
            print("🧠 No skills found.")

    def pulse(self):
        print(f"💓 [{self.node_id}] - Role: {self.role} - Pulse OK")
