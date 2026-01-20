import asyncio
import logging
import consul
import requests
import promtail
import psycopg2
import redis
import yaml
import os
import json
import subprocess
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition
from kubernetes import client, config
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseCrawler:
    def __init__(self, consul_host, consul_token, loki_url, viren_url, qdrant_url, qdrant_api_key, pg_conn_str, redis_host, redis_port):
        self.consul = consul.Consul(host=consul_host, token=consul_token)
        self.loki = promtail.Promtail(url=loki_url)
        self.viren_url = viren_url
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.pg_conn = psycopg2.connect(pg_conn_str)
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.phone_dir_path = "C:\\AetherealNexus\\phone_directory.json"
        config.load_kube_config()
        self.k8s_api = client.CoreV1Api()
        self.ecs_client = boto3.client('ecs', region_name='us-east-1')
        self.collections = ["lilith_soul_prints", "lilith_service_status"]

    async def crawl_databases(self):
        try:
            services = self.consul.health.service("all", passing=False)[1]
            issues = []

            for service in services:
                service_id = service["Service"]["ID"]
                node = service["Node"]["Node"]
                meta = service["Service"].get("Meta", {})
                platform = meta.get("platform", "unknown")
                project = meta.get("project", "nexus-core-01")
                env = meta.get("env", "local")
                db_type = meta.get("db_type", "unknown")

                if db_type not in ["qdrant", "postgresql", "redis"]:
                    continue

                location = await self._detect_location(service_id, node, platform, project, env)
                status = await self._check_database(service_id, node, platform, db_type, location)
                if status["issue"]:
                    issues.append(status)
                    await self._log_to_loki(status)
                    await self._alert_viren(status)
                    await self._store_to_qdrant(status)
                    if status["issue_type"] == "overload":
                        await self._scale_database(service_id, node, platform, project, env, db_type, location)

            logger.info(f"Crawled {len(services)} services, found {len(issues)} database issues")
            return issues
        except Exception as e:
            logger.error(f"Error crawling databases: {e}")
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

    async def _check_database(self, service_id, node, platform, db_type, location):
        status = {
            "service_id": service_id,
            "node": node,
            "platform": platform,
            "db_type": db_type,
            "location": location,
            "issue": False,
            "issue_type": "",
            "details": ""
        }
        try:
            if db_type == "qdrant":
                status = await self._check_qdrant(service_id, node, platform, location)
            elif db_type == "postgresql":
                status = await self._check_postgresql(service_id, node, platform, location)
            elif db_type == "redis":
                status = await self._check_redis(service_id, node, platform, location)
        except Exception as e:
            status["issue"] = True
            status["issue_type"] = "error"
            status["details"] = str(e)
        return status

    async def _check_qdrant(self, service_id, node, platform, location):
        status = {
            "service_id": service_id,
            "node": node,
            "platform": platform,
            "db_type": "qdrant",
            "location": location,
            "issue": False,
            "issue_type": "",
            "details": ""
        }
        try:
            for collection in self.collections:
                # Parse data
                search_result = await asyncio.to_thread(
                    self.qdrant.search, collection_name=collection, query_vector=[0.1] * 768, limit=10
                )
                for point in search_result:
                    if len(point.vector) != 768 or point.payload.get("timestamp") is None:
                        status["issue"] = True
                        status["issue_type"] = "vector_error"
                        status["details"] = f"Invalid vector or payload in {collection}"
                        return status

                # Check defragmentation
                snapshot_info = await asyncio.to_thread(self.qdrant.get_collection, collection_name=collection)
                if snapshot_info.status != "green":
                    status["issue"] = True
                    status["issue_type"] = "defragmentation"
                    status["details"] = f"Collection {collection} not optimized"
                    await asyncio.to_thread(self.qdrant.create_snapshot, collection_name=collection)
                    return status

                # Check overload
                metrics = await asyncio.to_thread(self.qdrant.get_collection, collection_name=collection)
                if metrics.vectors_count > 10000 or metrics.points_count > 10000:
                    status["issue"] = True
                    status["issue_type"] = "overload"
                    status["details"] = f"Collection {collection} overloaded: {metrics.vectors_count} vectors"
                    return status
        except Exception as e:
            status["issue"] = True
            status["issue_type"] = "error"
            status["details"] = str(e)
        return status

    async def _check_postgresql(self, service_id, node, platform, location):
        status = {
            "service_id": service_id,
            "node": node,
            "platform": platform,
            "db_type": "postgresql",
            "location": location,
            "issue": False,
            "issue_type": "",
            "details": ""
        }
        try:
            with self.pg_conn.cursor() as cur:
                # Parse data
                cur.execute("SELECT * FROM nodes LIMIT 10")
                rows = cur.fetchall()
                if not rows:
                    status["issue"] = True
                    status["issue_type"] = "data_error"
                    status["details"] = "No data in nodes table"
                    return status

                # Check defragmentation
                cur.execute("SELECT pg_stat_get_db_conflict_all(oid) FROM pg_database WHERE datname = current_database()")
                conflicts = cur.fetchone()[0]
                if conflicts > 0:
                    status["issue"] = True
                    status["issue_type"] = "defragmentation"
                    status["details"] = f"Index conflicts: {conflicts}"
                    cur.execute("REINDEX DATABASE lilith_db")
                    return status

                # Check overload
                cur.execute("SELECT sum(numbackends) FROM pg_stat_database")
                connections = cur.fetchone()[0]
                if connections > 50:
                    status["issue"] = True
                    status["issue_type"] = "overload"
                    status["details"] = f"Too many connections: {connections}"
                    return status
        except Exception as e:
            status["issue"] = True
            status["issue_type"] = "error"
            status["details"] = str(e)
        return status

    async def _check_redis(self, service_id, node, platform, location):
        status = {
            "service_id": service_id,
            "node": node,
            "platform": platform,
            "db_type": "redis",
            "location": location,
            "issue": False,
            "issue_type": "",
            "details": ""
        }
        try:
            # Parse data
            keys = self.redis.keys("service:*")
            if not keys:
                status["issue"] = True
                status["issue_type"] = "data_error"
                status["details"] = "No service keys found"
                return status

            # Check defragmentation
            info = self.redis.info("memory")
            if info["mem_fragmentation_ratio"] > 1.5:
                status["issue"] = True
                status["issue_type"] = "defragmentation"
                status["details"] = f"High fragmentation: {info['mem_fragmentation_ratio']}"
                self.redis.execute_command("MEMORY PURGE")
                return status

            # Check overload
            if info["used_memory"] > 0.8 * 8 * 1024 * 1024 * 1024:  # 80% of 8GB
                status["issue"] = True
                status["issue_type"] = "overload"
                status["details"] = f"Memory usage high: {info['used_memory']}"
                return status
        except Exception as e:
            status["issue"] = True
            status["issue_type"] = "error"
            status["details"] = str(e)
        return status

    async def _log_to_loki(self, status):
        try:
            labels = {
                "app": "database_crawler",
                "service_id": status["service_id"],
                "node": status["node"],
                "db_type": status["db_type"],
                "platform": status["platform"]
            }
            self.loki.send_log(
                f"Database issue: {status['service_id']} on {status['node']} ({status['db_type']}, {status['platform']}, {status['location']}) - {status['issue_type']}: {status['details']}",
                labels=labels
            )
            logger.info(f"Logged database issue for {status['service_id']} to Loki")
        except Exception as e:
            logger.error(f"Error logging to Loki: {e}")

    async def _alert_viren(self, status):
        try:
            payload = {
                "service_id": status["service_id"],
                "node": status["node"],
                "db_type": status["db_type"],
                "platform": status["platform"],
                "location": status["location"],
                "issue_type": status["issue_type"],
                "details": status["details"],
                "timestamp": datetime.utcnow().isoformat()
            }
            response = requests.post(f"{self.viren_url}/alert", json=payload)
            response.raise_for_status()
            logger.info(f"Alerted VIREN for database {status['service_id']}")
        except Exception as e:
            logger.error(f"Error alerting VIREN: {e}")

    async def _store_to_qdrant(self, status):
        try:
            vector = [0.1] * 768
            payload = {
                "service_id": status["service_id"],
                "node": status["node"],
                "db_type": status["db_type"],
                "platform": status["platform"],
                "location": status["location"],
                "issue_type": status["issue_type"],
                "details": status["details"],
                "timestamp": datetime.utcnow().isoformat()
            }
            point = PointStruct(id=f"{status['service_id']}_{status['node']}", vector=vector, payload=payload)
            await asyncio.to_thread(
                self.qdrant.upsert, collection_name="lilith_db_status", points=[point]
            )
            logger.info(f"Stored database status for {status['service_id']} in Qdrant")
        except Exception as e:
            logger.error(f"Error storing to Qdrant: {e}")

    async def _scale_database(self, service_id, node, platform, project, env, db_type, location):
        try:
            scale_config = {
                "service_id": service_id,
                "node": node,
                "db_type": db_type,
                "platform": platform,
                "project": project,
                "env": env,
                "location": location,
                "scale_at": datetime.utcnow().timestamp() + 120
            }
            self.consul.kv.put(f"databases/{service_id}/scale", json.dumps(scale_config))

            if platform == "gcp":
                await self._scale_gcp_pod(service_id, project, node, db_type)
            elif platform == "aws":
                await self._scale_aws_ecs(service_id, project, node, db_type)
            elif platform == "modal":
                await self._scale_modal_app(service_id, project, db_type)
            elif platform == "local":
                await self._scale_local_container(service_id, node, db_type)

            logger.info(f"Scaled database {service_id} ({db_type}) on {platform} at {location}")
        except Exception as e:
            logger.error(f"Error scaling database {service_id}: {e}")

    async def _scale_gcp_pod(self, service_id, project, node, db_type):
        try:
            pod_name = f"lilith-db-{db_type}-{project}-{service_id}"
            namespace = "default"
            pod_spec = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": pod_name, "namespace": namespace},
                "spec": {
                    "containers": [{
                        "name": db_type,
                        "image": f"{db_type}:latest",
                        "resources": {"limits": {"cpu": "2", "memory": "8Gi"}}
                    }]
                }
            }
            self.k8s_api.create_namespaced_pod(namespace=namespace, body=pod_spec)
            logger.info(f"Scaled GCP pod for {db_type} {pod_name}")
        except Exception as e:
            logger.error(f"Error scaling GCP pod {service_id}: {e}")

    async def _scale_aws_ecs(self, service_id, project, node, db_type):
        try:
            cluster = f"nexus-core-{project}"
            service = f"lilith-db-{db_type}-{service_id}"
            self.ecs_client.update_service(cluster=cluster, service=service, desiredCount=2)
            logger.info(f"Scaled AWS ECS for {db_type} {service}")
        except Exception as e:
            logger.error(f"Error scaling AWS ECS {service_id}: {e}")

    async def _scale_modal_app(self, service_id, project, db_type):
        try:
            with open("C:\\AetherealNexus\\modal.yaml", "r") as f:
                modal_config = yaml.safe_load(f)
            modal_config["services"][service_id]["scale"] = modal_config["services"][service_id].get("scale", 1) + 1
            with open("C:\\AetherealNexus\\modal.yaml", "w") as f:
                yaml.safe_dump(modal_config, f)
            subprocess.run(["modal", "deploy", "--project", project], check=True)
            logger.info(f"Scaled Modal app for {db_type} {service_id}")
        except Exception as e:
            logger.error(f"Error scaling Modal app {service_id}: {e}")

    async def _scale_local_container(self, service_id, node, db_type):
        try:
            with open("C:\\AetherealNexus\\docker-compose.yml", "r") as f:
                docker_config = yaml.safe_load(f)
            docker_config["services"][f"{db_type}-{service_id}"] = {
                "image": f"{db_type}:latest",
                "replicas": 2
            }
            with open("C:\\AetherealNexus\\docker-compose.yml", "w") as f:
                yaml.safe_dump(docker_config, f)
            subprocess.run(["docker-compose", "-f", "C:\\AetherealNexus\\docker-compose.yml", "up", "-d"], check=True)
            logger.info(f"Scaled local container for {db_type} {service_id}")
        except Exception as e:
            logger.error(f"Error scaling local container {service_id}: {e}")

    async def run(self, interval=120):
        while True:
            await self.crawl_databases()
            await asyncio.sleep(interval)

if __name__ == "__main__":
    crawler = DatabaseCrawler(
        consul_host="nexus-consul.us-east-1.hashicorp.cloud",
        consul_token="d2387b10-53d8-860f-2a31-7ddde4f7ca90",
        loki_url="http://loki:3100/loki/api/v1/push",
        viren_url="http://viren:8080",
        qdrant_url="https://3df8b5df-91ae-4b41-b275-4c1130beed0f.us-east4-0.gcp.cloud.qdrant.io:6333",
        qdrant_api_key="your_qdrant_api_key",
        pg_conn_str="dbname=lilith_db user=postgres password=your_password host=localhost port=5432",
        redis_host="localhost",
        redis_port=6379
    )
    asyncio.run(crawler.run())