import requests

class PsychAdapter:
    def __init__(self, base_url: str = "http://localhost:8018"):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=2.0)
        r.raise_for_status()
        return r.json()

    def report_emotion(self, emotion: str, intensity: float) -> dict:
        payload = {"operation": "emotional_state", "data": {"emotion": emotion, "intensity": float(intensity)}}
        r = requests.post(f"{self.base_url}/check_in", json=payload, timeout=2.0)
        r.raise_for_status()
        return r.json()

    def report_pain(self, level: float, context: str) -> dict:
        payload = {"operation": "report_pain", "data": {"level": float(level), "context": context}}
        r = requests.post(f"{self.base_url}/check_in", json=payload, timeout=2.0)
        r.raise_for_status()
        return r.json()
