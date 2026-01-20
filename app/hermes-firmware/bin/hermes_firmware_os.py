# hermes_os_final.py — Hermes Universal OS (Resilience + Auth + Guest + Firmware Agent)
# Notes:
# - Adds JWT auth (admin + guest), rate limiting, robust deploy w/ retries, compose v1/v2,
#   Docker image ensure/pull, LLM readiness checks, background task tracking, graceful shutdown,
#   Qdrant sync de-dup (stable IDs), firmware agent with cluster management, and improved logging.
# - Expose THIS app (PORT, default 8080) to the internet, but keep your llama-server at 127.0.0.1:11434.
# - Env you likely want:
#   SECRET_KEY, ADMIN_USER, ADMIN_PASSWORD_HASH, MODEL_PATH, LLM_URL,
#   QDRANT_URL, QDRANT_API_KEY, NATS_URL, PLATFORM, LITE_MODE, PORT

import os
import json
import asyncio
import logging
import uuid
import signal
import hashlib
import shutil
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List

import aiohttp
from aiohttp import ClientError, ClientTimeout
from fastapi import FastAPI, Request, HTTPException, status, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from docker import DockerClient
from docker.errors import DockerException, APIError, ImageNotFound, NotFound
import nats
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
from nats.aio.client import Client as NATSClient
import pathlib
from subprocess import CalledProcessError

# Auth & limiting
import time
import jwt
from passlib.hash import bcrypt
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------- logging ----------
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger  =  logging.getLogger("HermesOS")

# ---------- config ----------
class PlatformType(Enum):
    WINDOWS_VULKAN  =  "windows-vulkan"
    WINDOWS_CPU  =  "windows-cpu"
    LINUX_CUDA  =  "linux-cuda"
    LINUX_ROCM  =  "linux-rocm"
    LINUX_SYCL  =  "linux-sycl"
    MACOS_METAL  =  "macos-metal"
    CPU  =  "cpu"

@dataclass
class AppConfig:
    platform: PlatformType
    lite_mode: bool
    model_path: str
    llm_url: str
    qdrant_url: str
    qdrant_api_key: Optional[str]
    nats_url: str
    host: str
    port: int
    collection_name: str  =  "viren_tech"
    vector_size: int  =  768
    heartbeat_interval: int  =  5
    pulse_interval: int  =  30
    request_timeout: int  =  60
    # deployment resilience
    deploy_retries: int  =  4
    deploy_initial_delay: float  =  1.0
    deploy_max_delay: float  =  12.0
    # nats resilience
    nats_max_retries: int  =  5
    nats_initial_delay: float  =  2.0
    # firmware settings
    firmware_base_path: str  =  "hermes-firmware/bin"

# ---------- auth ----------
JWT_ALG  =  "HS256"
SECRET  =  os.getenv("SECRET_KEY", "dev-secret-change-me")
ADMIN_USER  =  os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD_HASH  =  os.getenv("ADMIN_PASSWORD_HASH", bcrypt.hash("admin"))  # dev default

GUEST_RPM  =  int(os.getenv("GUEST_RPM", "60"))
GUEST_BURST  =  int(os.getenv("GUEST_BURST", "10"))

def make_token(sub: str, role: str, ttl_minutes: int  =  120) -> str:
    now  =  int(time.time())
    payload  =  {"sub": sub, "role": role, "iat": now, "exp": now + ttl_minutes * 60}
    return jwt.encode(payload, SECRET, algorithm = JWT_ALG)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms = [JWT_ALG])

# ---------- firmware agent ----------
class FirmwareAgent:
    def __init__(self, base_path: str = "hermes-firmware/bin"):
        self.base_path = pathlib.Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.clusters = ["cluster1", "cluster2", "cluster3", "cluster4", "cluster5"]
        self._ensure_cluster_dirs()
        
        # Hardware Control
        self.cpu_cores  =  psutil.cpu_count(logical = False)
        self.assigned_cores  =  {}  # agent_name -> [core_ids]
        self.model_database_map  =  {}  # 2 models : 1 database

        # Traffic Management
        self.throughput_monitor  =  {}
        self.priority_queues  =  {}
    
    def _ensure_cluster_dirs(self):
        """Create cluster directories if they don't exist"""
        for cluster in self.clusters:
            cluster_path = self.base_path / cluster
            cluster_path.mkdir(parents=True, exist_ok=True)
    
    def get_cluster_path(self, cluster: str) -> pathlib.Path:
        """Get path for a specific cluster"""
        if cluster not in self.clusters:
            raise ValueError(f"Invalid cluster: {cluster}. Must be one of {self.clusters}")
        return self.base_path / cluster
    
    def list_firmware_files(self, cluster: str) -> List[Dict]:
        """List all firmware files in a cluster"""
        cluster_path = self.get_cluster_path(cluster)
        files = []
        for file_path in cluster_path.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "cluster": cluster
                })
        return files
    
    def upload_firmware(self, cluster: str, file: UploadFile, filename: Optional[str] = None) -> bool:
        """Upload firmware file to specific cluster"""
        try:
            cluster_path = self.get_cluster_path(cluster)
            filename = filename or file.filename
            if not filename:
                return False
            
            file_path = cluster_path / filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Uploaded firmware {filename} to {cluster}")
            return True
        except Exception as e:
            logger.error(f"Firmware upload failed: {e}")
            return False
    
    def delete_firmware(self, cluster: str, filename: str) -> bool:
        """Delete firmware file from cluster"""
        try:
            cluster_path = self.get_cluster_path(cluster)
            file_path = cluster_path / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted firmware {filename} from {cluster}")
                return True
            return False
        except Exception as e:
            logger.error(f"Firmware deletion failed: {e}")
            return False
    
    def get_firmware_info(self, cluster: str, filename: str) -> Optional[Dict]:
        """Get detailed information about a firmware file"""
        try:
            cluster_path = self.get_cluster_path(cluster)
            file_path = cluster_path / filename
            if file_path.exists():
                stat = file_path.stat()
                return {
                    "name": filename,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "cluster": cluster,
                    "path": str(file_path)
                }
            return None
        except Exception as e:
            logger.error(f"Firmware info retrieval failed: {e}")
            return None

    def pin_agent_to_cpu(self, agent_name: str, core_ids: List[int]):
        """Pin agent process to specific CPU cores"""
        try:
            # Get agent process (simplified - would need actual PID)
            agent_process = self._find_agent_process(agent_name)
            if agent_process:
                agent_process.cpu_affinity(core_ids)
                self.assigned_cores[agent_name] = core_ids
                logger.info(f"Pinned {agent_name} to cores {core_ids}")
                return True
        except Exception as e:
            logger.error(f"CPU pinning failed for {agent_name}: {e}")
        return False

    def _find_agent_process(self, agent_name: str):
        """Find process by agent name (simplified implementation)"""
        # This is a simplified implementation
        # In production, you'd need proper process tracking
        try:
            current_process = psutil.Process()
            return current_process
        except:
            return None

    def allocate_model_database_pair(self, model1: str, model2: str, db_config: Dict):
        """Allocate database for 2 models following 2:1 ratio"""
        pair_id = f"{model1}_{model2}"
        self.model_database_map[pair_id] = {
            "models": [model1, model2],
            "database": db_config,
            "sync_status": "pending",
            "throughput": 0
        }
        logger.info(f"Database allocated for {model1} + {model2}")
        return pair_id

    def optimize_hardware_traffic(self):
        """Dynamically optimize hardware resource allocation"""
        current_load = psutil.cpu_percent(percpu=True)
        memory_usage = psutil.virtual_memory().percent

        # Rebalance CPU assignments based on load
        for agent_name, cores in self.assigned_cores.items():
            core_loads = [current_load[core] for core in cores]
            avg_load = sum(core_loads) / len(core_loads)

            if avg_load > 80:  # Overloaded cores
                # Find less loaded cores
                available_cores = [i for i in range(self.cpu_cores)
                             if current_load[i] < 50]
                if available_cores:
                    new_cores = available_cores[:len(cores)]
                    self.pin_agent_to_cpu(agent_name, new_cores)

        # Monitor database throughput
        for pair_id, pair_info in self.model_database_map.items():
            throughput = self._measure_database_throughput(pair_id)
            pair_info["throughput"] = throughput

    def _measure_database_throughput(self, pair_id: str) -> float:
        """Measure database operations per second"""
        return len(self.model_database_map.get(pair_id, {}).get("recent_ops", []))

    async def health_check(self) -> Dict:
        cpu_load = psutil.cpu_percent(percpu=True)
        memory_usage = psutil.virtual_memory().percent

        # Count firmware files across all clusters
        total_firmware_files = 0
        for cluster in self.clusters:
            cluster_path = self.get_cluster_path(cluster)
            if cluster_path.exists():
                total_firmware_files += len([f for f in cluster_path.iterdir() if f.is_file()])

        return {
            "agent": "firmware_manager",
            "status": "active",
            "cpu_cores_total": self.cpu_cores,
            "cpu_cores_assigned": len(self.assigned_cores),
            "average_cpu_load": sum(cpu_load) / len(cpu_load),
            "memory_usage_percent": memory_usage,
            "model_database_pairs": len(self.model_database_map),
            "firmware_stats": {
                "total_files": total_firmware_files,
                "clusters": self.clusters,
                "base_path": str(self.base_path)
            }
        }

# ---------- main class ----------
class HermesOS:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client: Optional[DockerClient] = None
        self.qdrant: Optional[QdrantClient] = None
        self.nc: Optional[NATSClient] = None
        self.containers: Dict[str, Any] = {}
        self.app = FastAPI(title=f"Hermes-{config.platform.value}")
        self._shutdown_event = asyncio.Event()
        self.background_tasks: List[asyncio.Task] = []
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # Initialize firmware agent
        self.firmware_agent = FirmwareAgent(config.firmware_base_path)

        # API
        self._setup_api()

        # Initialize dependencies (non-lite)
        if not self.config.lite_mode:
            self._initialize_docker()
            self._initialize_qdrant()
            # nats is async
            self.background_tasks.append(asyncio.create_task(self._initialize_nats()))

        # FastAPI lifecycle: create shared HTTP session
        @self.app.on_event("startup")
        async def _on_startup():
            self._http_session = aiohttp.ClientSession()

        @self.app.on_event("shutdown")
        async def _on_shutdown():
            if self._http_session:
                await self._http_session.close()
                self._http_session = None

    # ---------- init helpers ----------
    def _initialize_docker(self):
        try:
            self.client = DockerClient.from_env()
            self.client.ping()
            logger.info("Docker initialized")
        except DockerException as e:
            logger.warning(f"Docker init failed: {e}")
            self.client = None

    def _initialize_qdrant(self):
        try:
            self.qdrant = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key, timeout=10)
            try:
                self.qdrant.get_collection(self.config.collection_name)
            except (UnexpectedResponse, ValueError):
                self.qdrant.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=models.VectorParams(size=self.config.vector_size, distance=models.Distance.COSINE),
                )
            logger.info("Qdrant initialized")
        except Exception as e:
            logger.error(f"Qdrant init failed: {e}")
            self.qdrant = None

    async def _initialize_nats(self):
        max_retries = self.config.nats_max_retries
        delay = self.config.nats_initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                self.nc = await nats.connect(
                    servers=[self.config.nats_url],
                    max_reconnect_attempts=5,
                    reconnect_time_wait=2,
                    name=f"hermes-{self.config.platform.value}",
                )
                logger.info("NATS connected")
                await self.nc.subscribe("viren.ui.core.req", cb=self._handle_request)
                # heartbeat task
                self.background_tasks.append(asyncio.create_task(self._nats_heartbeat()))
                return
            except (ErrConnectionClosed, ErrTimeout, ErrNoServers) as e:
                logger.warning(f"NATS attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
        logger.error("NATS failed after retries")
        self.nc = None

    # ---------- docker & deploy ----------
    async def _ensure_image(self, image: str) -> bool:
        """Ensure docker image exists; try pull if not present."""
        if not self.client:
            return False
        loop = asyncio.get_running_loop()
        try:
            # Check local images
            imgs = await loop.run_in_executor(None, self.client.images.list)
            if any(image in (t for img in imgs for t in img.tags)):  # image:tag match
                return True
        except DockerException as e:
            logger.warning(f"Image list failed: {e}")
        # Try pull
        try:
            logger.info(f"Pulling image {image}...")
            await loop.run_in_executor(None, lambda: self.client.images.pull(image))
            return True
        except (ImageNotFound, APIError, DockerException) as e:
            logger.error(f"Image pull failed for {image}: {e}")
            return False

    async def start_llm_container(self) -> bool:
        if not self.client:
            logger.warning("Docker unavailable (lite mode or daemon down)")
            return False

        container_name = f"hermes-{self.config.platform.value}"
        image = (
            f"hermes-{self.config.platform.value}:latest"
            if self.config.platform != PlatformType.CPU
            else "hermes-cpu:latest"
        )

        # Ensure image present (pull if needed)
        if not await self._ensure_image(image):
            return False

        loop = asyncio.get_running_loop()
        # Remove dead container if exists
        try:
            existing = await loop.run_in_executor(None, lambda: self.client.containers.get(container_name))
            if existing.status == "running":
                self.containers[container_name] = existing
                logger.info(f"Container {container_name} already running")
                return True
            await loop.run_in_executor(None, existing.remove, True)
        except NotFound:
            pass
        except DockerException as e:
            logger.warning(f"While removing old container {container_name}: {e}")

        model_dir = str(pathlib.Path(self.config.model_path).parent)
        env = {
            "MODEL_PATH": self.config.model_path,
            "NGL": "-1" if self.config.platform != PlatformType.CPU else "0",
        }

        try:
            container = await loop.run_in_executor(
                None,
                lambda: self.client.containers.run(
                    image,
                    "/run.sh",
                    name=container_name,
                    cpus=2.0,
                    mem_limit="4g",
                    environment=env,
                    ports={"11434/tcp": 11434},
                    detach=True,
                    volumes={model_dir: {"bind": "/models", "mode": "ro"}},
                    restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
                ),
            )
            self.containers[container_name] = container
            logger.info(f"Started container {container_name}")
            return True
        except (DockerException, APIError, ImageNotFound) as e:
            logger.error(f"Container start failed: {e}")
            return False

    async def deploy_platform(self) -> bool:
        if self.config.lite_mode:
            logger.info("Lite mode: skipping deployment")
            return False

        async def _run_once(cmd: List[str]) -> bool:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                out, err = await proc.communicate()
                if proc.returncode == 0:
                    logger.info(f"Command OK: {' '.join(cmd)}")
                    return True
                logger.error(f"Command failed ({' '.join(cmd)}): {err.decode(errors='ignore')}")
                return False
            except FileNotFoundError:
                logger.error(f"Command not found: {' '.join(cmd)}")
                return False

        # Map scripted platforms
        script_map = {
            PlatformType.WINDOWS_VULKAN: ["powershell", "-File", "windows/build-windows-vulkan.ps1"],
            PlatformType.WINDOWS_CPU: ["cmd", "/c", "windows/run-windows-cpu.cmd"],
            PlatformType.LINUX_CUDA: ["bash", "linux/build-linux-cuda.sh"],
            PlatformType.LINUX_ROCM: ["bash", "linux/build-linux-rocm.sh"],
            PlatformType.LINUX_SYCL: ["bash", "linux/build-linux-sycl.sh"],
            PlatformType.MACOS_METAL: ["bash", "macos/build-macos-metal.sh"],
        }

        delay = self.config.deploy_initial_delay
        for attempt in range(1, self.config.deploy_retries + 1):
            logger.info(f"Deploy attempt {attempt}/{self.config.deploy_retries} for {self.config.platform.value}")

            ok = False
            if self.config.platform in script_map:
                ok = await _run_once(script_map[self.config.platform])
                if ok:
                    ok = await self.start_llm_container()
            elif self.config.platform == PlatformType.CPU:
                # Support compose v1 and v2
                commands = [
                    ["docker-compose", "-f", "docker/compose-cpu.yaml", "up", "-d"],
                    ["docker", "compose", "-f", "docker/compose-cpu.yaml", "up", "-d"],
                ]
                for cmd in commands:
                    if await _run_once(cmd):
                        ok = True
                        break
            else:
                logger.error(f"Unsupported platform: {self.config.platform.value}")

            # LLM readiness gate
            if ok and await self._wait_llm_ready("deploy"):
                return True

            await asyncio.sleep(delay)
            delay = min(delay * 2.0, self.config.deploy_max_delay)

        logger.error("Deployment failed after retries")
        return False

    # ---------- readiness ----------
    async def _llm_ready(self) -> bool:
        url = f"{self.config.llm_url.rstrip('/')}/v1/models"
        try:
            if self._http_session is None:
                # Early boot: use a temporary session
                async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as s:
                    async with s.get(url) as resp:
                        return resp.status == 200
            else:
                async with self._http_session.get(url, timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _wait_llm_ready(self, label: str, timeout: int = 90) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if await self._llm_ready():
                logger.info(f"LLM ready after {label}")
                return True
            await asyncio.sleep(2)
        logger.error(f"LLM not ready within {timeout}s ({label})")
        return False

    # ---------- crawl & qdrant ----------
    @staticmethod
    def _stable_id(path: str, size: int) -> str:
        h = hashlib.sha1()
        h.update(path.encode("utf-8"))
        h.update(str(size).encode("utf-8"))
        return h.hexdigest()

    async def crawl_tech(self, repo_path: str):
        if not self.qdrant:
            logger.warning("Qdrant unavailable, skipping crawl")
            return
        loop = asyncio.get_running_loop()
        try:
            files = await loop.run_in_executor(
                None, lambda: list(pathlib.Path(repo_path).glob("**/*.gguf"))
            )
            points = []
            for f in files:
                if not await loop.run_in_executor(None, f.is_file):
                    continue
                size = await loop.run_in_executor(None, lambda: f.stat().st_size)
                pid = self._stable_id(str(f), size)
                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=[0.0] * self.config.vector_size,
                        payload={
                            "filename": f.name,
                            "path": str(f),
                            "size": size,
                            "platform": self.config.platform.value,
                        },
                    )
                )
            if points:
                self.qdrant.upsert(collection_name=self.config.collection_name, points=points, wait=True)
                logger.info(f"Crawled {len(points)} model files -> Qdrant")
        except Exception as e:
            logger.error(f"Crawl failed: {e}")

    async def sync_models_to_qdrant(self):
        """Query LLM for /v1/models and sync payload to Qdrant (best-effort)."""
        if not (self.qdrant and await self._llm_ready()):
            return
        try:
            async with (self._http_session or aiohttp.ClientSession()) as s:
                async with s.get(f"{self.config.llm_url.rstrip('/')}/v1/models") as r:
                    if r.status != 200:
                        return
                    data = await r.json()
                    models_list = data.get("models", [])
                    upserts = []
                    for m in models_list:
                        name = m.get("name") or m.get("model") or m.get("id")
                        size = int(m.get("size") or 0) if isinstance(m.get("size"), (int, str)) else 0
                        pid = self._stable_id(str(name), size)
                        upserts.append(
                            models.PointStruct(
                                id=pid,
                                vector=[0.0] * self.config.vector_size,
                                payload={"llm_model": name, "platform": self.config.platform.value, "raw": m},
                            )
                        )
                    if upserts:
                        self.qdrant.upsert(self.config.collection_name, upserts, wait=False)
        except Exception as e:
            logger.warning(f"sync_models_to_qdrant skipped: {e}")

    # ---------- pulse & heartbeat ----------
    async def pulse(self):
        while not self._shutdown_event.is_set():
            try:
                if self.client:
                    for name, c in list(self.containers.items()):
                        try:
                            await asyncio.get_running_loop().run_in_executor(None, c.reload)
                            if c.status != "running":
                                logger.warning(f"Container {name} not running (status = {c.status})")
                        except (APIError, NotFound) as e:
                            logger.error(f"Container {name} error: {e}")
                            self.containers.pop(name, None)
                if self.qdrant:
                    try:
                        self.qdrant.get_collections()
                    except Exception as e:
                        logger.error(f"Qdrant health check failed: {e}")
                # Optional: check LLM readiness and sync models
                if await self._llm_ready():
                    await self.sync_models_to_qdrant()
                # Optimize hardware traffic
                self.firmware_agent.optimize_hardware_traffic()
            except Exception as e:
                logger.error(f"Pulse error: {e}")
            await asyncio.sleep(self.config.pulse_interval)

    async def _nats_heartbeat(self):
        while not self._shutdown_event.is_set() and self.nc and self.nc.is_connected:
            try:
                hb = {
                    "id": f"hermes-{self.config.platform.value}",
                    "ok": True,
                    "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                await self.nc.publish("health.viren.core", json.dumps(hb).encode())
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)

    # ---------- API ----------
    def _setup_api(self):
        app = self.app
        # CORS for public UI (tighten in prod)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Rate limiter
        limiter = Limiter(key_func=get_remote_address, default_limits=[])
        app.state.limiter = limiter

        @app.exception_handler(RateLimitExceeded)
        async def _rl_handler(request, exc):
            return PlainTextResponse("Rate limit exceeded", status_code=429)

        # Auth helpers
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        auth_scheme = HTTPBearer(auto_error=False)

        def current_user(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)) -> dict:
            if not creds or not creds.credentials:
                raise HTTPException(status_code=401, detail="Missing token")
            try:
                return decode_token(creds.credentials)
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")

        def require_role(role: str):
            def _inner(user = Depends(current_user)):
                if user.get("role") != role:
                    raise HTTPException(status_code=403, detail="Forbidden")
                return user
            return _inner

        # ---- AUTH ROUTES ----
        @app.post("/auth/login")
        async def login(data: dict):
            username = (data or {}).get("username", "")
            password = (data or {}).get("password", "")
            if username != ADMIN_USER or not bcrypt.verify(password, ADMIN_PASSWORD_HASH):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            return {"access_token": make_token(sub=username, role="admin"), "token_type": "bearer"}

        @app.post("/auth/guest")
        @limiter.limit(f"{GUEST_RPM}/minute;{GUEST_BURST}/1second")
        async def guest():
            return {"access_token": make_token(sub="guest", role="guest", ttl_minutes=30), "token_type": "bearer"}

        @app.get("/auth/me")
        async def me(user = Depends(current_user)):
            return {"sub": user["sub"], "role": user["role"], "exp": user["exp"]}

        # ---- PUBLIC INFO ----
        @app.get("/health")
        async def health_check():
            status_obj = {
                "platform": self.config.platform.value,
                "status": "healthy",
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "components": {},
            }
            loop = asyncio.get_running_loop()
            if self.containers:
                for name, container in self.containers.items():
                    try:
                        await loop.run_in_executor(None, container.reload)
                        status_obj["components"][name] = container.status
                        if container.status != "running":
                            status_obj["status"] = "degraded"
                    except Exception as e:
                        status_obj["components"][name] = f"error: {e}"
                        status_obj["status"] = "unhealthy"
            if self.qdrant:
                try:
                    self.qdrant.get_collections()
                    status_obj["components"]["qdrant"] = "healthy"
                except Exception as e:
                    status_obj["components"]["qdrant"] = f"error: {e}"
                    status_obj["status"] = "unhealthy"
            if self.nc:
                status_obj["components"]["nats"] = "connected" if self.nc.is_connected else "disconnected"
                if not self.nc.is_connected:
                    status_obj["status"] = "degraded"
            # LLM
            status_obj["components"]["llm"] = "ready" if await self._llm_ready() else "init"
            # Firmware agent
            firmware_health = await self.firmware_agent.health_check()
            status_obj["components"]["firmware_agent"] = firmware_health
            return status_obj

        # ---- MODELS: proxy if available ----
        @app.get("/v1/models")
        async def list_models():
            if await self._llm_ready():
                try:
                    async with (self._http_session or aiohttp.ClientSession()) as s:
                        async with s.get(f"{self.config.llm_url.rstrip('/')}/v1/models") as r:
                            return await r.json()
                except Exception:
                    pass
            return {"models": [{"id": "hermes-7b", "platform": self.config.platform.value}]}

        # ---- CHAT: auth (guest or admin), rate limited ----
        @app.post("/v1/chat/completions")
        @limiter.limit(f"{GUEST_RPM}/minute;{GUEST_BURST}/1second")
        async def chat_completions(request: Request, user = Depends(current_user)):
            try:
                data = await request.json()
                async with (self._http_session or aiohttp.ClientSession()) as s:
                    async with s.post(
                        f"{self.config.llm_url.rstrip('/')}/v1/chat/completions",
                        json=data,
                        timeout=ClientTimeout(total=self.config.request_timeout),
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        raise HTTPException(status_code=resp.status, detail=await resp.text())
            except ClientError as e:
                raise HTTPException(status_code=503, detail=f"LLM error: {e}")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="LLM timeout")

        # ---- FIRMWARE MANAGEMENT ROUTES ----
        @app.get("/firmware/clusters")
        async def list_clusters(user = Depends(current_user)):
            """List all available clusters"""
            return {"clusters": self.firmware_agent.clusters}

        @app.get("/firmware/{cluster}/files")
        async def list_firmware_files(cluster: str, user = Depends(current_user)):
            """List all firmware files in a cluster"""
            try:
                files = self.firmware_agent.list_firmware_files(cluster)
                return {"cluster": cluster, "files": files}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.post("/firmware/{cluster}/upload")
        async def upload_firmware(
            cluster: str,
            file: UploadFile = File(...),
            filename: Optional[str] = Form(None),
            user = Depends(require_role("admin"))
        ):
            """Upload firmware file to specific cluster"""
            try:
                success = self.firmware_agent.upload_firmware(cluster, file, filename)
                if success:
                    return {"status": "success", "message": f"File uploaded to {cluster}"}
                else:
                    raise HTTPException(status_code=500, detail="Upload failed")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/firmware/{cluster}/download/{filename}")
        async def download_firmware(cluster: str, filename: str, user = Depends(current_user)):
            """Download firmware file from cluster"""
            try:
                cluster_path = self.firmware_agent.get_cluster_path(cluster)
                file_path = cluster_path / filename
                if file_path.exists():
                    return FileResponse(
                        path=file_path,
                        filename=filename,
                        media_type='application/octet-stream'
                    )
                else:
                    raise HTTPException(status_code=404, detail="File not found")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.delete("/firmware/{cluster}/files/{filename}")
        async def delete_firmware(cluster: str, filename: str, user = Depends(require_role("admin"))):
            """Delete firmware file from cluster"""
            try:
                success = self.firmware_agent.delete_firmware(cluster, filename)
                if success:
                    return {"status": "success", "message": f"File {filename} deleted from {cluster}"}
                else:
                    raise HTTPException(status_code=404, detail="File not found")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/firmware/{cluster}/files/{filename}/info")
        async def get_firmware_info(cluster: str, filename: str, user = Depends(current_user)):
            """Get information about a firmware file"""
            try