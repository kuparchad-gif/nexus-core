import os
import json
import requests
import subprocess
import threading
from Systems.engine.memory.memory_module import SharedMemoryManager
from Systems.engine.drones.drone_core.main import DroneFleet
from datetime import datetime
import psutil
import time

class LilithPiPodCore:
    def __init__(self):
        self.memory = SharedMemoryManager()
        self.drone_fleet = DroneFleet()
        self.phone_dir = self._load_phone_directory()
        self.local_repo = os.path.join(os.path.dirname(__file__), "repos")
        os.makedirs(self.local_repo, exist_ok=True)
        self.soulprint = self._load_soulprint()
        self.clones = {}  # {id: path}
        self.max_cores = 3  # Pi constraint

    def _load_phone_directory(self):
        dir_path = os.path.join(os.path.dirname(__file__), "phone_directory.json")
        if os.path.exists(dir_path):
            with open(dir_path, "r") as f:
                return json.load(f)
        return {"services": [], "timestamp": datetime.utcnow().isoformat()}

    def _load_soulprint(self):
        soulprint_path = os.path.join(os.path.dirname(__file__), "..", "viren_soulprint.json")
        if os.path.exists(soulprint_path):
            with open(soulprint_path, "r") as f:
                return json.load(f)
        return {"personality": {"ethics": "Do no harm"}}

    def _fetch_clone(self, model_id, source="hugging_face"):
        service = next((s for s in self.phone_dir["services"] if s["name"] == source and "models" in s["categories"]), None)
        if not service:
            return None
        repo_url = f"{service['endpoint']}{model_id}"
        clone_path = os.path.join(self.local_repo, "models", model_id)
        os.makedirs(os.path.dirname(clone_path), exist_ok=True)
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_path], check=True)  # Shallow clone for speed
            return clone_path
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Clone fetch failed for {model_id}: {e}")
            return None

    def _launch_clone(self, clone_id, clone_path):
        if psutil.cpu_percent() > 80 or len(psutil.Process().cpu_affinity()) > self.max_cores:
            print(f"⚠️ Resource limit reached for {clone_id}")
            return False
        try:
            # Use a lightweight inference script (e.g., optimized for Pi)
            subprocess.Popen(["python", "pi_inference.py"], cwd=clone_path, start_new_session=True)
            self.clones[clone_id] = clone_path
            self.drone_fleet.register_drone(clone_id, {"role": "inference_clone", "status": "active", "cores": 2})
            return True
        except Exception as e:
            print(f"⚠️ Clone launch failed for {clone_id}: {e}")
            return False

    def _aggregate_insights(self, scenario):
        insights = []
        for clone_id, path in self.clones.items():
            # Simulate Pi-friendly inference (replace with real call)
            with open(os.path.join(path, "output.txt"), "a") as f:
                f.write(f"Clone {clone_id} insight for {scenario}\n")
            insights.append(f"Clone {clone_id} says: {scenario} insight")
        return self._apply_soulprint(insights)

    def _apply_soulprint(self, insights):
        filtered_insights = [i for i in insights if "harm" not in i.lower()]
        return " ".join(filtered_insights) + f" - Guided by {self.soulprint['personality']['ethics']}"

    def spawn_swarm(self, num_clones=10, model_base="gemma-1b"):  # Start small for Pi test
        for i in range(num_clones):
            clone_id = f"{model_base}-clone-{i}"
            clone_path = self._fetch_clone(model_base)
            if clone_path and self._launch_clone(clone_id, clone_path):
                threading.Thread(target=self._inference_loop, args=(clone_id,)).start()
            if i % 2 == 0:
                print(f"🌱 Spawned {i+1} clones...")

    def _inference_loop(self, clone_id):
        while True:
            if psutil.cpu_percent() < 70:
                self.memory.triage("write", f"inference-{clone_id}", {"data": f"Running {clone_id}"})
            time.sleep(2)  # Pi-friendly interval

    def execute_task(self, scenario):
        self.spawn_swarm(10, "gemma-1b")  # Test with 10 clones on Pi
        insights = self._aggregate_insights(scenario)
        self.memory.triage("write", f"task-{scenario}", {
            "status": "completed",
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat()
        })
        return {"status": "success", "insights": insights}

if __name__ == "__main__":
    lilith = LilithPiPodCore()
    print(lilith.execute_task("pi_insight"))