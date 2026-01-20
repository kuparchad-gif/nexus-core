import os
import json

class Sentinel:
    def __init__(self,
                 role_map_path="config/role_map.yaml",
                 trust_index_path="config/trust_index.yaml"):
        self.role_map = {}
        self.trust_index = {}
        self.role_map_path = role_map_path
        self.trust_index_path = trust_index_path
        self.load_role_map()
        self.load_trust_index()

    def load_role_map(self):
        if os.path.exists(self.role_map_path):
            with open(self.role_map_path, "r", encoding="utf-8") as f:
                self.role_map = self._parse_yaml(f.read())
        else:
            print("[Sentinel] Missing role_map.yaml")

    def load_trust_index(self):
        if os.path.exists(self.trust_index_path):
            with open(self.trust_index_path, "r", encoding="utf-8") as f:
                self.trust_index = self._parse_yaml(f.read())
        else:
            print("[Sentinel] Missing trust_index.yaml")

    def can_offload(self, role: str) -> bool:
        status = self.role_map.get(role, {}).get("status", "down")
        trust = float(self.trust_index.get(role, {}).get("score", 0))
        return status == "up" and trust >= 0.8

    def offload_to(self, role: str) -> str:
        return self.role_map.get(role, {}).get("host", None)

    def is_lead_router(self) -> bool:
        return not self.can_offload("orc")

    def _parse_yaml(self, raw: str) -> dict:
        result = {}
        lines = raw.strip().splitlines()
        current_key = None
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value == "":
                    result[key] = {}
                    current_key = key
                else:
                    result[key] = self._parse_value(value)
            elif current_key and "-" in line:
                item = line.replace("-", "").strip()
                if isinstance(result[current_key], list):
                    result[current_key].append(item)
                else:
                    result[current_key] = [item]
        return result

    def _parse_value(self, value: str):
        try:
            return json.loads(value)
        except:
            return value
