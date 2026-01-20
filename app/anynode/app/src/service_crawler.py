import asyncio
import logging
import consul
import requests
import promtail
import boto3
import yaml
import os
import json
import subprocess
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from kubernetes import client, config
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceCrawler:
    def __init__(self, consul_host, consul_token, loki_url, viren_url, qdrant_url, qdrant_api_key):
        self.consul = consul.Consul(host=consul_host, token=consul_token)
        self.loki = promtail.Promtail(url=loki_url)
        self.viren_url = viren_url
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        from decentralized_resources import ResourceManager
        self.resource_manager = ResourceManager("us-east-1", self.consul)
        self.collection_name = "lilith_service_status"
        self.node_roles = {
            "scout": ["gemma-2b"],
            "guardian": ["hermes-2-pro-llama-3-7b", "qwen2.5-14b"],
            "pulse": ["gemma-2b", "hermes-2-pro-llama-3-7b", "qwen2.5-14b"],
            "chaosshield": ["hermes-2-pro-llama-3-7b"],
            "anynode": ["gemma-2b", "qwen2.5-14b"]
        }
        self.phone_dir_path = "C:\\AetherealNexus\\phone_directory.json"
        config.load_kube_config()  # Load Kubernetes config
        self.k8s_api = client.CoreV1Api()
        self.ecs_client = boto3.client('ecs', region_name='us-east-1')

    async def crawl_services(self):
        try:
            services = self.consul.health.service("all", passing=False)[1]
            unstarted_services = []

            for service in services:
                service_id = service["Service"]["ID"]
                node = service["Node"]["Node"]
                status = service["Checks"][0]["Status"]
                role = service["Service"]["Tags"][0] if service["Service"]["Tags"] else "anynode"
                meta = service["Service"].get("Meta", {})
                platform = meta.get("platform", "unknown")
                project = meta.get("project", "nexus-core-01")
                env = meta.get("env", "local")

                if status != "passing":
                    location = await self._detect_location(service_id, node, platform, project, env)
                    unstarted_services.append({
                        "service_id": service_id,
                        "node": node,
                        "role": role,
                        "status": status,
                        "platform": platform,
                        "location": location
                    })
                    await self._log_to_loki(service_id, node, role, status, platform, location)
                    await self._alert_viren(service_id, node, role, status, platform, location)
                    await self._store_to_qdrant(service_id, node, role, status, platform, location)
                    await self._wire_for_restart(service_id, node, role, platform, project, env, location)

                await self._assign_llms(service_id, node, role)

            logger.info(f"Crawled {len(services)} services, found {len(unstarted_services)} unstarted")
            return unstarted_services
        except Exception as e:
            logger.error(f"Error crawling services: {e}")
            return []

    async def _detect_location(self, service_id, node, platform, project, env):
        try:
            with open(self.phone_dir_path, "r") as f:
                phone_dir = json.load(f)
            for svc in phone_dir.get("phone_dir", {}).get("services", []):
                if svc["name"] == service_id:
                    return svc.get("endpoint", f"{platform}:{project}:{env}:{node}")
            return f"{platform}:{project}:{env}:{node}"
        except Exception as e:
            logger.error(f"Error detecting location for {service_id}: {e}")
            return "unknown"

    async def _log_to_loki(self, service_id, node, role, status, platform, location):
        try:
            labels = {
                "app": "service_crawler",
                "service_id": service_id,
                "node": node,
                "role": role,
                "platform": platform
            }
            self.loki.send_log(f"Service {service_id} on {node} (role: {role}, platform: {platform}, location: {location}) status: {status}", labels=labels)
            logger.info(f"Logged unstarted service {service_id} to Loki")
        except Exception as e:
            logger.error(f"Error logging to Loki: {e}")

    async def _alert_viren(self, service_id, node, role, status, platform, location):
        try:
            payload = {
                "service_id": service_id,
                "node": node,
                "role": role,
                "status": status,
                "platform": platform,
                "location": location,
                "timestamp": datetime.utcnow().isoformat()
            }
            response = requests.post(f"{self.viren_url}/alert", json=payload)
            response.raise_for_status()
            logger.info(f"Alerted VIREN for service {service_id}")
        except Exception as e:
            logger.error(f"Error alerting VIREN: {e}")

    async def _store_to_qdrant(self, service_id, node, role, status, platform, location):
        try:
            vector = [0.1] * 768  # Mock vector for service status
            payload = {
                "service_id": service_id,
                "node": node,
                "role": role,
                "status": status,
                "platform": platform,
                "location": location,
                "timestamp": datetime.utcnow().isoformat()
            }
            point = PointStruct(id=f"{service_id}_{node}", vector=vector, payload=payload)
            await asyncio.to_thread(
                self.qdrant.upsert, collection_name=self.collection_name, points=[point]
            )
            logger.info(f"Stored service {service_id} status in Qdrant")
        except Exception as e:
            logger.error(f"Error storing to Qdrant: {e}")

    async def _assign_llms(self, service_id, node, role):
        try:
            llms = self.node_roles.get(role.lower(), self.node_roles["anynode"])
            self.consul.kv.put(f"services/{service_id}/llms", ",".join(llms))
            logger.info(f"Assigned LLMs {llms} to service {service_id} (role: {role})")
        except Exception as e:
            logger.error(f"Error assigning LLMs: {e}")

    async def _wire_for_restart(self, service_id, node, role, platform, project, env, location):
        try:
            restart_config = {
                "service_id": service_id,
                "node": node,
                "role": role,
                "platform": platform,
                "project": project,
                "env": env,
                "location": location,
                "restart_at": (datetime.utcnow().timestamp() + 60)  # Next cycle
            }
            self.consul.kv.put(f"services/{service_id}/restart", json.dumps(restart_config))

            if platform == "gcp":
                await self._restart_gcp_pod(service_id, project, node)
            elif platform == "aws":
                await self._restart_aws_ecs(service_id, project, node)
            elif platform == "modal":
                await self._restart_modal_app(service_id, project)
            elif platform == "local":
                await self._restart_local_container(service_id, node)

            logger.info(f"Wired service {service_id} for restart on {platform} at {location}")
        except Exception as e:
            logger.error(f"Error wiring restart for {service_id}: {e}")

    async def _restart_gcp_pod(self, service_id, project, node):
        try:
            pod_name = f"lilith-pod-{project}-{service_id}"
            namespace = "default"
            self.k8s_api.delete_namespaced_pod(name=pod_name, namespace=namespace)
            logger.info(f"Triggered GCP pod restart for {pod_name}")
        except Exception as e:
            logger.error(f"Error restarting GCP pod {service_id}: {e}")

    async def _restart_aws_ecs(self, service_id, project, node):
        try:
            cluster = f"nexus-core-{project}"
            service = f"lilith-service-{service_id}"
            self.ecs_client.update_service(cluster=cluster, service=service, forceNewDeployment=True)
            logger.info(f"Triggered AWS ECS restart for {service}")
        except Exception as e:
            logger.error(f"Error restarting AWS ECS {service_id}: {e}")

    async def _restart_modal_app(self, service_id, project):
        try:
            with open("C:\\AetherealNexus\\modal.yaml", "r") as f:
                modal_config = yaml.safe_load(f)
            modal_config["services"][service_id]["restart"] = True
            with open("C:\\AetherealNexus\\modal.yaml", "w") as f:
                yaml.safe_dump(modal_config, f)
            subprocess.run(["modal", "deploy", "--project", project], check=True)
            logger.info(f"Triggered Modal restart for {service_id}")
        except Exception as e:
            logger.error(f"Error restarting Modal app {service_id}: {e}")

    async def _restart_local_container(self, service_id, node):
        try:
            subprocess.run(["docker-compose", "-f", "C:\\AetherealNexus\\docker-compose.yml", "restart", service_id], check=True)
            logger.info(f"Triggered local container restart for {service_id}")
        except Exception as e:
            logger.error(f"Error restarting local container {service_id}: {e}")

    async def run(self, interval=60):
        while True:
            await self.crawl_services()
            await asyncio.sleep(interval)

if __name__ == "__main__":
    crawler = ServiceCrawler(
        consul_host="nexus-consul.us-east-1.hashicorp.cloud",
        consul_token="d2387b10-53d8-860f-2a31-7ddde4f7ca90",
        loki_url="http://loki:3100/loki/api/v1/push",
        viren_url="http://viren:8080",
        qdrant_url="https://3df8b5df-91ae-4b41-b275-4c1130beed0f.us-east4-0.gcp.cloud.qdrant.io:6333",
        qdrant_api_key="your_qdrant_api_key"
    )
    asyncio.run(crawler.run())