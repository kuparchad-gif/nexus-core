import psutil
import requests
import time
import consul
import os
import uuid
import subprocess
import json
from datetime import datetime
from soulprint import generate_soulprint, verify_soulprint, quarantine_node

class LilithSwarmCore:
    def __init__(self):
        self.node_id = os.getenv("NODE_ID", f"node-{uuid.uuid4().hex[:8]}")
        self.node_type = os.getenv("NODE_TYPE", "node")
        self.project = os.getenv("PROJECT", "nexus-core-01")
        self.env = os.getenv("ENVIRONMENT", "local")
        self.phone_dir = self._load_phone_directory()
        self.soulprint = self._load_soulprint()
        self.happiness = 75
        self.health = 85
        self.last_load_time = 0
        self.clones = {}
        self.max_clones = 10  # Free-tier limit
        self.node_roles = {
            "pulse": ["gemma-2b", "hermes-2-pro-llama-3-7b"],  # Limit to 2 LLMs for 8GB RAM
            "scout": ["gemma-2b"],
            "guardian": ["gemma-2b", "hermes-2-pro-llama-3-7b"],
            "chaosshield": ["gemma-2b"],
            "anynode": ["gemma-2b", "hermes-2-pro-llama-3-7b"]
        }
        self.local_repo = os.path.join(os.path.dirname(__file__), "repos")
        os.makedirs(self.local_repo, exist_ok=True)

    def _load_phone_directory(self):
        dir_path = os.path.join(os.path.dirname(__file__), "phone_directory.json")
        if os.path.exists(dir_path):
            with open(dir_path, "r") as f:
                return json.load(f)
        return {"services": [], "timestamp": datetime.utcnow().isoformat()}

    def _load_soulprint(self):
        soulprint_path = os.path.join(os.path.dirname(__file__), "viren_soulprint.json")
        if os.path.exists(soulprint_path):
            with open(soulprint_path, "r") as f:
                return json.load(f)
        return {"personality": {"ethics": "Do no harm"}}

    def _fetch_clone(self, model_id, source="hugging_face"):
        service = next((s for s in self.phone_dir["services"] if s["name"] == source and "models" in s.get("categories", [])), None)
        if not service:
            return None
        repo_url = f"{service['endpoint']}{model_id}"
        clone_path = os.path.join(self.local_repo, "models", model_id)
        os.makedirs(os.path.dirname(clone_path), exist_ok=True)
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_path], check=True)
            return clone_path
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Clone fetch failed for {model_id}: {e}")
            return None

    def _launch_clone(self, clone_id, clone_path):
        if psutil.cpu_percent() > 80 or len(self.clones) >= self.max_clones:
            print(f"⚠️ Resource or clone limit reached for {clone_id}")
            return False
        try:
            subprocess.Popen(["python", "inference.py"], cwd=clone_path, start_new_session=True)
            self.clones[clone_id] = clone_path
            return True
        except Exception as e:
            print(f"⚠️ Clone launch failed for {clone_id}: {e}")
            return False

    def _aggregate_insights(self, scenario):
        insights = []
        for clone_id in self.clones:
            insight = f"Clone {clone_id} says: {scenario} insight {datetime.now().microsecond}"
            insights.append(insight)
        return self._apply_soulprint(insights)

    def _apply_soulprint(self, insights):
        filtered_insights = [i for i in insights if "harm" not in i.lower()]
        return " ".join(filtered_insights) + f" - Guided by {self.soulprint['personality']['ethics']}"

    def spawn_swarm(self, num_clones=10, model_base="gemma-1b"):
        for i in range(min(num_clones, self.max_clones)):
            clone_id = f"{model_base}-clone-{i}"
            clone_path = self._fetch_clone(model_base)
            if clone_path and self._launch_clone(clone_id, clone_path):
                print(f"🌱 Spawned clone {clone_id}")
            if i % 5 == 0:
                print(f"🌱 Spawned {i+1} clones...")

    def get_metrics(self):
        """Get CPU/memory usage."""
        return {"cpu": psutil.cpu_percent(interval=0.1), "mem": psutil.virtual_memory().percent}

    def find_anynode(self):
        """Find ANYNODE via Consul."""
        c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
        for _, nodes in c.catalog.services()[1].items():
            for node in c.catalog.service(nodes[0])[1]:
                if node.get('ServiceMeta', {}).get('type') == 'anynode':
                    return node['ServiceAddress']
        return None

    def transfer_task(self, task):
        """Send task to ANYNODE."""
        anynode = self.find_anynode()
        if anynode:
            try:
                r = requests.post(f"http://{anynode}:8081/api/task/transfer", json=task, timeout=5)
                return r.status_code == 200
            except:
                return False
        return False

    def run_envoy(self):
        """Start Envoy proxy for ANYNODE."""
        envoy_config = """
static_resources:
  listeners:
  - address:
      socket_address:
        address: 0.0.0.0
        port_value: 8081
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: backend
          http_filters:
          - name: envoy.filters.http.router
  clusters:
  - name: backend
    connect_timeout: 5s
    type: LOGICAL_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: backend
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: localhost
                port_value: 5000
"""
        with open("/tmp/envoy.yaml", "w") as f:
            f.write(envoy_config)
        subprocess.Popen(["envoy", "-c", "/tmp/envoy.yaml"])

    def report_status(self):
        """Update happiness and health based on metrics."""
        metrics = self.get_metrics()
        self.happiness = max(50, self.happiness + (metrics["cpu"] < 80 and 1 or -1))
        self.health = max(50, self.health + (metrics["mem"] < 90 and 1 or -1))
        requests.post("http://localhost:5000/api/config/update", json={
            "node_id": self.node_id,
            "metrics": metrics,
            "happiness": self.happiness,
            "health": self.health
        })

    def main(self):
        """Node loop with soulprint verification and swarm logic."""
        manifest, signature = generate_soulprint(self.node_id, self.project, self.env)
        if not verify_soulprint(self.node_id, manifest, signature):
            quarantine_node(self.node_id)
            raise SystemExit("Soulprint verification failed")

        c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
        port = {"anynode": 8081, "soulsync": 8082, "nexpulse": 8083, "chaosshield": 8084}.get(self.node_type, 5000)

        if self.node_type == "anynode":
            self.run_envoy()

        c.agent.service.register(
            name=f"lilith-{self.node_id}", service_id=self.node_id, address=f"{self.node_id}.local", port=port,
            meta={"type": self.node_type, "project": self.project, "env": self.env}
        )

        if self.node_type == "node":
            self.spawn_swarm(num_clones=10, model_base="gemma-1b")

        while True:
            self.phone_dir = self._load_phone_directory()
            metrics = self.get_metrics()
            c.agent.service.register(
                name=f"lilith-{self.node_id}", service_id=self.node_id, address=f"{self.node_id}.local", port=port,
                meta={"type": self.node_type, "cpu": str(metrics["cpu"]), "mem": str(metrics["mem"]), "project": self.project, "env": self.env}
            )
            self.report_status()
            if self.node_type == "node" and metrics["cpu"] > 80:
                self.transfer_task({"type": "process_chunk", "data": "chunk", "source": self.node_id})
            time.sleep(10)

if __name__ == "__main__":
    lilith = LilithSwarmCore()
    lilith.main()