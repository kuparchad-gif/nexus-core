# valhalla/orchestrator.py
import docker
import time
import requests
import redis
from . import registry

class CloudOrchestrator:
    """Orchestrates the deployment of services to cloud nodes."""
    
    def __init__(self, provider):
        self.provider = provider
        self.docker_client = docker.from_env()
        
    def deploy_core_services(self, nodes):
        """Deploys Qdrant, Redis, and LoRA Engine to the given nodes."""
        print("🚀 Deploying core services to provisioned nodes...")
        
        # In a real scenario, we'd deploy to each node.
        # For this simulation, we'll deploy them locally as a stand-in.
        self._deploy_qdrant()
        self._deploy_redis_mesh()
        self._deploy_lora_engine()
        
        print("✅ Core services deployed.")
        
        if self.validate_services():
            # Register services after they are healthy
            registry.register_service("memory_cluster", "localhost:6333")
            registry.register_service("lora_inference", "localhost:7860")
            return True
        return False

    def _deploy_qdrant(self, image="qdrant/qdrant:v1.7.4", name="qdrant_memlayer"):
        print("  -> Deploying Qdrant memory layer...")
        try:
            self.docker_client.containers.run(
                image, detach=True, name=name, ports={'6333/tcp': 6333}
            )
        except docker.errors.ContainerError as e:
            if "already in use" in e.stderr:
                print("     Qdrant container already running.")
            else:
                raise e
    
    def _deploy_redis_mesh(self, count=3, base_port=6380):
        print("  -> Deploying Redis cache mesh...")
        for i in range(count):
            name = f"redis_cache_{i+1}"
            port = base_port + i
            try:
                self.docker_client.containers.run(
                    "redis:7-alpine", detach=True, name=name, ports={f'{port}/tcp': 6379}
                )
            except docker.errors.ContainerError as e:
                if "already in use" in e.stderr:
                    print(f"     Redis container {name} already running.")
                else:
                    raise e

    def _deploy_lora_engine(self, image="ghcr.io/huggingface/text-generation-inference:latest", name="lora_inference"):
        print("  -> Deploying LoRA inference engine...")
        try:
            self.docker_client.containers.run(
                image, detach=True, name=name, ports={'7860/tcp': 7860},
                command=["--model-id", "microsoft/phi-2", "--quantize", "nf4"]
            )
        except docker.errors.ContainerError as e:
            if "already in use" in e.stderr:
                print("     LoRA container already running.")
            else:
                raise e

    def validate_services(self, timeout=60):
        """Validates that all deployed services are healthy."""
        print("🔬 Validating service health...")
        # Qdrant
        if not self._wait_for_service("Qdrant", "http://localhost:6333", timeout):
            return False
        # Redis
        for i in range(3):
            if not self._wait_for_redis(6380 + i, timeout):
                return False
        # LoRA
        if not self._wait_for_service("LoRA", "http://localhost:7860/health", timeout):
            return False
            
        print("✅ All services are healthy.")
        return True

    def deploy_spirallaspan_node(self, name="spirallaspan_cloud_permanent"):
        """Builds and deploys the Spirallaspan container."""
        print("  -> Building and deploying Spirallaspan Node...")
        try:
            # Build the image from the Dockerfile
            self.docker_client.images.build(path=".", dockerfile="Spirallaspan.Dockerfile", tag="spirallaspan:latest")
            
            # Run the container
            self.docker_client.containers.run(
                "spirallaspan:latest", detach=True, name=name, ports={'8080/tcp': 8080}
            )
            print(f"✅ Spirallaspan container {name} started.")
            return True
        except docker.errors.ContainerError as e:
            if "already in use" in e.stderr:
                print("     Spirallaspan container already running.")
                return True
            else:
                raise e

    def _deploy_lillith_core(self, name="lillith_core"):
        print("  -> Building and deploying Lillith Core...")
        try:
            # Build the image from the Dockerfile
            self.docker_client.images.build(path=".", dockerfile="LillithCore.Dockerfile", tag="lillith_core:latest")
            
            # Run the container
            self.docker_client.containers.run(
                "lillith_core:latest", detach=True, name=name, ports={'8001/tcp': 8001}
            )
        except docker.errors.ContainerError as e:
            if "already in use" in e.stderr:
                print("     Lillith Core container already running.")
            else:
                raise e

    def _wait_for_service(self, name, url, timeout):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if requests.get(url).status_code == 200:
                    print(f"  ✅ {name} is healthy.")
                    return True
            except requests.ConnectionError:
                time.sleep(1)
        print(f"  ❌ {name} did not become healthy in time.")
        return False
        
    def _wait_for_redis(self, port, timeout):
        r = redis.Redis(port=port)
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if r.ping():
                    print(f"  ✅ Redis at port {port} is healthy.")
                    return True
            except redis.ConnectionError:
                time.sleep(1)
        print(f"  ❌ Redis at port {port} did not become healthy in time.")
        return False

if __name__ == "__main__":
    from .scouts import CloudScout
    scout = CloudScout()
    resources = scout.discover_resources()
    nodes = [scout.provision_node(res) for res in resources]
    
    orchestrator = CloudOrchestrator(scout)
    orchestrator.deploy_core_services(nodes)
