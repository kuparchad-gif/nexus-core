# metatron_router.py
import modal
from typing import Dict, List, Tuple
import json
import asyncio
import numpy as np
import psutil
import ping3
import clamd
import zlib
import hmac
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import base64
import os
import uuid
from datetime import datetime
from functools import lru_cache
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

# Import pulse components (from anynode.doc)
from Systems.engine.pulse.pulse_core import start_pulse_system, PULSE_SOURCES, ACTIVE_SOURCES_LOCK
from Systems.engine.pulse.pulse_listener import PulseListener
from Systems.address_manager.pulse13 import NexusAddress
from Utilities.network_core.quantum_stream import QuantumStream

# NEW: Health predictor
class HealthPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.training_data = []

    def update(self, cpu: float, mem: float, latency: float, io: float, score: float):
        self.training_data.append([cpu, mem, latency, io, score])
        if len(self.training_data) > 100:
            X = np.array([[d[0], d[1], d[2], d[3]] for d in self.training_data[-100:]])
            y = np.array([d[4] for d in self.training_data[-100:]])
            self.model.fit(X, y)

    def predict(self, cpu: float, mem: float, latency: float, io: float) -> float:
        if len(self.training_data) < 10:
            return None
        return float(self.model.predict([[cpu, mem, latency, io]])[0])

# NEW: Quantum walk routing with sparse matrix
def quantum_walk_route(nodes: List[Dict], query_load: int, media_type: str) -> List[Dict]:
    """Quantum-inspired routing using sparse matrix"""
    def build_adjacency_matrix(nodes: List[Dict]) -> csr_matrix:
        n = len(nodes)
        data, row_ind, col_ind = [], [], []
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                health_diff = abs(node_i["health"]["health_score"] - node_j["health"]["health_score"])
                tenant_match = 1.0 if node_i["tags"][0] == node_j["tags"][0] else 0.5
                weight = node_i["health"]["health_score"] * tenant_match / (1 + health_diff)
                data.append(weight)
                row_ind.append(i)
                col_ind.append(j)
        adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
        row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        return adj_matrix / row_sums[:, np.newaxis]

    def quantum_walk(adj_matrix: csr_matrix, steps: int = 5) -> np.ndarray:
        n = adj_matrix.shape[0]
        state = np.ones(n) / n
        for _ in range(min(steps, 5 + n // 100)):
            state = adj_matrix.dot(state)
        return state / state.sum()

    assignments = []
    adj_matrix = build_adjacency_matrix(nodes)
    probabilities = quantum_walk(adj_matrix)
    
    batch_size = 10
    for i in range(0, query_load, batch_size):
        batch_count = min(batch_size, query_load - i)
        node_indices = np.random.choice(len(nodes), size=batch_count, p=probabilities)
        for j, node_idx in enumerate(node_indices):
            node = nodes[node_idx]
            assignments.append({
                "task_id": f"task-{uuid.uuid4()}",
                "target_node": node["address"],
                "nexus_address": node["nexus_address"],
                "health_score": node["health"]["health_score"],
                "media_type": media_type,
                "quantum_weight": float(probabilities[node_idx])
            })
    return assignments

@lru_cache(maxsize=10)
def consciousness_grid(size: int) -> Tuple[np.ndarray, Dict]:
    """Cached Ulam spiral grid"""
    grid = np.zeros((size, size), dtype=int)
    x, y = size // 2, size // 2
    num = 1
    step = 1
    direction = 0
    steps_taken = 0
    steps_to_take = 1
    signal_map = {}
    
    while num <= size * size:
        grid[x, y] = num
        signal_map[num] = (x, y)
        num += 1
        if direction == 0:  # Right
            y += 1
        elif direction == 1:  # Up
            x -= 1
        elif direction == 2:  # Left
            y -= 1
        elif direction == 3:  # Down
            x += 1
        steps_taken += 1
        if steps_taken == steps_to_take:
            steps_taken = 0
            direction = (direction + 1) % 4
            if direction % 2 == 0:
                steps_to_take += 1
    return grid, signal_map

def route_with_prime_weights(grid: np.ndarray, query_load: int) -> List[Dict]:
    """Fallback Ulam routing"""
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    size = grid.shape[0]
    assignments = []
    for i in range(query_load):
        task_id = f"task-{uuid.uuid4()}"
        prime_weight = 1.0 if is_prime(i + 1) else 0.5
        grid_pos = (i % size, (i // size) % size)
        assignments.append({
            "task_id": task_id,
            "grid_position": grid_pos,
            "prime_weight": prime_weight
        })
    return assignments

image = modal.Image.debian_slim().pip_install([
    "numpy==1.24.3",
    "psutil==5.9.5",
    "ping3==4.0.3",
    "python-clamd==0.4.0",
    "cryptography==41.0.7",
    "scipy==1.10.1",
    "scikit-learn==1.3.0"
]).apt_install(["clamav"])

app = modal.App("metatron-router")
volume = modal.Volume.from_name("cognikube-data", create_if_missing=True)

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=1800,
    secrets=[modal.Secret.from_name("cognikube-secrets")],
    volumes={"/cognikube": volume}
)
async def route_consciousness(size: int = 15, query_load: int = 100, media_type: str = "application/json", use_quantum: bool = True) -> Dict:
    """Hyper-optimized quantum-inspired router"""
    aes_key = os.getenv("AES_KEY", "12345678901234567890123456789012").encode()
    hmac_key = os.getenv("HMAC_KEY", "hmac-secret-1234567890").encode()
    
    class AESCipher:
        def __init__(self, key: bytes):
            self.key = key
            self.aesgcm = AESGCM(self.key)
            self.nonce_cache = {}

        def encrypt(self, plaintext: str) -> str:
            nonce = os.urandom(12)
            ciphertext = self.aesgcm.encrypt(nonce, plaintext.encode(), None)
            self.nonce_cache[base64.b64encode(ciphertext).decode()] = nonce
            return base64.b64encode(ciphertext).decode()

        def decrypt(self, ciphertext: str) -> str:
            ciphertext_bytes = base64.b64decode(ciphertext.encode())
            nonce = self.nonce_cache.get(ciphertext, os.urandom(12))
            plaintext = self.aesgcm.decrypt(nonce, ciphertext_bytes, None)
            return plaintext.decode()
    
    cipher = AESCipher(aes_key)
    
    pulse_listener = PulseListener()
    pulse_listener.start()
    
    quantum_stream = QuantumStream()
    
    try:
        clamav = clamd.ClamdUnixSocket()
        clamav.ping()
    except Exception as e:
        print(f"⚠️ ClamAV initialization failed: {e}")
        clamav = None
    
    health_predictor = HealthPredictor()
    
    tenant_tag = os.getenv("TENANT_TAG", "modal-default")
    node_freq = os.getenv("ANYNODE_FREQ", "13")
    nexus_address = NexusAddress(
        region=int(os.getenv("REGION_ID", "1")),
        node_type=int(os.getenv("NODE_TYPE", "1")),
        role_id=int(os.getenv("ROLE_ID", "1")),
        unit_id=int(os.getenv("UNIT_ID", "1"))
    )
    
    class DiscoveryService:
        def __init__(self):
            self.node_id = f"node-{uuid.uuid4()}"
            self.service_name = "lillith-node"
            self.address = nexus_address
            self.blocked_pulses = []
            self.task_success_rates: Dict[str, float] = {}
            self.health_cache: Dict[str, Dict] = {}
            self.scan_cache: Dict[str, float] = {}
            self.ip_cache: set = set()
            self.pulse_buffer: List[Dict] = []
            self._stop_health_monitor = asyncio.Event()

        async def health_monitor(self):
            while not self._stop_health_monitor.is_set():
                cache_key = f"{self.node_id}:{node_freq}:{tenant_tag}"
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                latency = ping3.ping("metatron-router.modal.run", unit="ms") or 1000
                disk_io = psutil.disk_io_counters().write_count if psutil.disk_io_counters() else 0
                success_rate = self.task_success_rates.get(self.node_id, 1.0)
                health_score = health_predictor.predict(cpu_usage, memory_usage, latency, disk_io)
                if health_score is None:
                    health_score = (
                        0.35 * (1 - cpu_usage / 100) +
                        0.35 * (1 - memory_usage / 100) +
                        0.15 * (1 - min(latency / 1000, 1)) +
                        0.05 * (1 - min(disk_io / 1000, 1)) +
                        0.1 * success_rate
                    )
                health = {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "latency_ms": latency,
                    "disk_io_ops": disk_io,
                    "success_rate": success_rate,
                    "health_score": health_score
                }
                health_predictor.update(cpu_usage, memory_usage, latency, disk_io, health_score)
                self.health_cache[cache_key] = {"health": health, "timestamp": datetime.now().timestamp()}
                await asyncio.sleep(5)

        async def prune_nodes(self):
            while not self._stop_health_monitor.is_set():
                with ACTIVE_SOURCES_LOCK:
                    stale = [s for s, d in PULSE_SOURCES.items() if time.time() - d["last_seen"] > 30]
                    for source in stale:
                        del PULSE_SOURCES[source]
                await asyncio.sleep(10)

        async def flush_pulse_buffer(self):
            while not self._stop_health_monitor.is_set():
                if self.pulse_buffer:
                    quanta = quantum_stream.split_memory_into_quanta(self.pulse_buffer)
                    quantum_stream.store_quanta(f"pulse_batch_{uuid.uuid4()}", quanta)
                    self.pulse_buffer.clear()
                await asyncio.sleep(5)

        async def register_node(self, host: str, port: int, freq: str, tenant: str):
            try:
                cache_key = f"{self.node_id}:{freq}:{tenant}"
                if cache_key in self.health_cache and (datetime.now().timestamp() - self.health_cache[cache_key]["timestamp"]) < 13:
                    health = self.health_cache[cache_key]["health"]
                else:
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    latency = ping3.ping("metatron-router.modal.run", unit="ms") or 1000
                    disk_io = psutil.disk_io_counters().write_count if psutil.disk_io_counters() else 0
                    success_rate = self.task_success_rates.get(self.node_id, 1.0)
                    health_score = health_predictor.predict(cpu_usage, memory_usage, latency, disk_io)
                    if health_score is None:
                        health_score = (
                            0.35 * (1 - cpu_usage / 100) +
                            0.35 * (1 - memory_usage / 100) +
                            0.15 * (1 - min(latency / 1000, 1)) +
                            0.05 * (1 - min(disk_io / 1000, 1)) +
                            0.1 * success_rate
                        )
                    health = {
                        "cpu_usage": cpu_usage,
                        "memory_usage": memory_usage,
                        "latency_ms": latency,
                        "disk_io_ops": disk_io,
                        "success_rate": success_rate,
                        "health_score": health_score
                    }
                    health_predictor.update(cpu_usage, memory_usage, latency, disk_io, health_score)
                    self.health_cache[cache_key] = {"health": health, "timestamp": datetime.now().timestamp()}
                
                pulse_data = {
                    "type": "pulse",
                    "source": self.node_id,
                    "timestamp": datetime.now().isoformat(),
                    "address": self.address.to_dict(),
                    "tenant": tenant,
                    "freq": freq,
                    "health": health
                }
                pulse_json = json.dumps(pulse_data)
                if clamav and pulse_json not in self.scan_cache:
                    try:
                        scan_result = await asyncio.to_thread(clamav.instream, pulse_json.encode())
                        if scan_result["status"] == "FOUND":
                            print(f"⚠️ Malicious pulse detected: {scan_result['stream']}")
                            return
                        self.scan_cache[pulse_json] = datetime.now().timestamp()
                    except Exception as e:
                        print(f"⚠️ ClamAV scan failed: {e}")
                
                encrypted_pulse = cipher.encrypt(pulse_json)
                pulse_signature = hmac.new(hmac_key, pulse_json.encode(), hashlib.sha256).hexdigest()
                payload = json.dumps({"encrypted": encrypted_pulse, "signature": pulse_signature})
                compressed = zlib.compress(payload.encode('utf-8'))
                
                broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                broadcast_socket.sendto(compressed, ('<broadcast>', 7777))
                broadcast_socket.close()
                
                self.pulse_buffer.append(pulse_data)
                print(f"✅ Node {self.node_id} pulsed (tenant: {tenant}, freq: {freq}, health: {health['health_score']:.2f})")
            except Exception as e:
                print(f"❌ Pulse registration failed: {e}")
            
        async def discover_nodes(self, tenant_filter: str = None, freq_filter: str = None) -> List[Dict]:
            async def evaluate_node(source: str, data: Dict) -> Dict:
                if time.time() - data["last_seen"] > 30:
                    return None
                if not self._validate_source(data["address"]):
                    self.blocked_pulses.append({"source": source, "address": data["address"], "timestamp": datetime.now().isoformat()})
                    quantum_stream.store_quanta(f"blocked_pulse_{source}", self.blocked_pulses[-1])
                    return None
                tags = [f"tenant:{data.get('tenant', tenant_tag)}", f"freq:{data.get('freq', node_freq)}"]
                if tenant_filter and f"tenant:{tenant_filter}" not in tags:
                    return None
                if freq_filter and f"freq:{freq_filter}" not in tags:
                    return None
                health = data.get("health", {"cpu_usage": 0, "memory_usage": 0, "latency_ms": 1000, "disk_io_ops": 0, "success_rate": 1.0, "health_score": 1.0})
                return {
                    "id": source,
                    "address": data["address"],
                    "port": 7777,
                    "tags": tags,
                    "status": "healthy",
                    "timestamp": data["timestamp"],
                    "nexus_address": data.get("address", {}),
                    "health": health
                }
            
            with ACTIVE_SOURCES_LOCK:
                tasks = [evaluate_node(source, data) for source, data in PULSE_SOURCES.items()]
                nodes = [node for node in await asyncio.gather(*tasks) if node]
                return sorted(nodes, key=lambda x: x["health"]["health_score"], reverse=True)
    
        def _validate_source(self, address: str) -> bool:
            if address in self.ip_cache:
                return True
            allowed_subnets = [
                "10.0.0.0/8",
                "172.16.0.0/12",
                "192.168.0.0/16"
            ]
            from ipaddress import ip_address, ip_network
            try:
                ip = ip_address(address)
                if any(ip in ip_network(subnet) for subnet in allowed_subnets):
                    self.ip_cache.add(address)
                    return True
                return False
            except:
                return False
    
    discovery_service = DiscoveryService()
    
    asyncio.create_task(discovery_service.health_monitor())
    asyncio.create_task(discovery_service.prune_nodes())
    asyncio.create_task(discovery_service.flush_pulse_buffer())
    
    await discovery_service.register_node(
        host="metatron-router.modal.run",
        port=443,
        freq=node_freq,
        tenant=tenant_tag
    )
    
    start_pulse_system()
    
    nodes = await discovery_service.discover_nodes(tenant_filter=None, freq_filter=node_freq)
    node_count = len(nodes)
    print(f"Routing {media_type} across {node_count} healthy nodes (freq: {node_freq})")
    
    if media_type not in ["application/json", "text/plain"] and clamav:
        try:
            payload = json.dumps({"grid_size": size, "query_load": query_load}).encode()
            if payload.decode() not in discovery_service.scan_cache:
                scan_result = await asyncio.to_thread(clamav.instream, payload)
                if scan_result["status"] == "FOUND":
                    raise ValueError(f"Malware detected: {scan_result['stream']}")
                discovery_service.scan_cache[payload.decode()] = datetime.now().timestamp()
        except Exception as e:
            print(f"⚠️ Antivirus scan failed: {e}")
    
    if use_quantum and len(nodes) > 1:
        assignments = quantum_walk_route(nodes, query_load, media_type)
    else:
        grid_data = consciousness_grid(size)
        assignments = route_with_prime_weights(grid_data[0], query_load)
        for i, node in enumerate(nodes[:len(assignments)]):
            if i < len(assignments):
                health = node["health"]
                weights = {
                    "application/json": {"cpu": 0.35, "mem": 0.35, "latency": 0.15, "io": 0.05, "success": 0.1},
                    "image/png": {"cpu": 0.3, "mem": 0.4, "latency": 0.2, "io": 0.05, "success": 0.05},
                    "video/mp4": {"cpu": 0.25, "mem": 0.3, "latency": 0.3, "io": 0.1, "success": 0.05},
                    "audio/mpeg": {"cpu": 0.3, "mem": 0.3, "latency": 0.25, "io": 0.1, "success": 0.05}
                }.get(media_type, {"cpu": 0.35, "mem": 0.35, "latency": 0.15, "io": 0.05, "success": 0.1})
                health_score = (
                    weights["cpu"] * (1 - health["cpu_usage"] / 100) +
                    weights["mem"] * (1 - health["memory_usage"] / 100) +
                    weights["latency"] * (1 - min(health["latency_ms"] / 1000, 1)) +
                    weights["io"] * (1 - min(health["disk_io_ops"] / 1000, 1)) +
                    weights["success"] * health["success_rate"]
                )
                assignments[i]["target_node"] = node["address"]
                assignments[i]["nexus_address"] = node["nexus_address"]
                assignments[i]["health_score"] = health_score
                assignments[i]["media_type"] = media_type
    
    quanta = quantum_stream.split_memory_into_quanta({
        "grid_size": size,
        "query_load": query_load,
        "assignments": assignments,
        "discovered_nodes": node_count,
        "timestamp": datetime.now().isoformat(),
        "media_type": media_type,
        "routing_mode": "quantum" if use_quantum else "ulam"
    })
    quantum_stream.store_quanta(f"route_{uuid.uuid4()}", quanta)
    
    return {
        "grid": consciousness_grid(size)[0].tolist() if not use_quantum else [],
        "assignments": assignments,
        "signal_map": consciousness_grid(size)[1] if not use_quantum else {},
        "discovered_nodes": nodes,
        "media_type": media_type,
        "routing_mode": "quantum" if use_quantum else "ulam"
    }