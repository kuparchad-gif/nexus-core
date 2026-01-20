# metatron_router.py
import modal
from typing import Dict, List, Tuple, Optional
import json
import asyncio
import numpy as np
import psutil
import ping3
import time  
import socket  
from ipaddress import ip_address, ip_network  
import zlib
import hmac
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import os
import uuid
from datetime import datetime
from functools import lru_cache
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

# Fixed AESCipher class
class AESCipher:
    def __init__(self, key: bytes):
        if len(key) not in [16, 24, 32]:
            raise ValueError("AES key must be 16, 24, or 32 bytes long")
        self.key = key
        self.aesgcm = AESGCM(self.key)
    
    def encrypt(self, plaintext: str) -> Tuple[str, str]:
        """Encrypt and return (ciphertext, nonce) as base64 strings"""
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext.encode(), None)
        return base64.b64encode(ciphertext).decode(), base64.b64encode(nonce).decode()
    
    def decrypt(self, ciphertext: str, nonce: str) -> str:
        """Decrypt using ciphertext and nonce (both base64 strings)"""
        ciphertext_bytes = base64.b64decode(ciphertext.encode())
        nonce_bytes = base64.b64decode(nonce.encode())
        plaintext = self.aesgcm.decrypt(nonce_bytes, ciphertext_bytes, None)
        return plaintext.decode()

# Health predictor
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

    def predict(self, cpu: float, mem: float, latency: float, io: float) -> Optional[float]:
        if len(self.training_data) < 10:
            return None
        return float(self.model.predict([[cpu, mem, latency, io]])[0])

# Quantum walk routing with sparse matrix
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

# Simple content validator instead of ClamAV
class ContentValidator:
    def __init__(self):
        self.suspicious_patterns = [
            b"eval(base64_decode",
            b"system($_GET",
            b"shell_exec",
            b"php_uname",
            b"/bin/bash"
        ]
    
    def validate_content(self, data: str) -> bool:
        """Basic content validation"""
        data_bytes = data.encode() if isinstance(data, str) else data
        for pattern in self.suspicious_patterns:
            if pattern in data_bytes:
                return False
        return True

image = modal.Image.debian_slim().pip_install([
    "numpy==1.24.3",
    "psutil==5.9.5",
    "ping3==4.0.3",
    "cryptography==41.0.7",
    "scipy==1.10.1",
    "scikit-learn==1.3.0"
])

app = modal.App("metatron-router")
volume = modal.Volume.from_name("cognikube-data", create_if_missing=True)

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=1800,
    secrets=[modal.Secret.from_name("cognikube-secrets")],
    volumes={"/data": volume}  # Fixed volume path
)
async def route_consciousness(size: int = 15, query_load: int = 100, media_type: str = "application/json", use_quantum: bool = True) -> Dict:
    """Hyper-optimized quantum-inspired router"""
    aes_key = os.getenv("AES_KEY", "12345678901234567890123456789012").encode()
    hmac_key = os.getenv("HMAC_KEY", "hmac-secret-1234567890").encode()
    
    cipher = AESCipher(aes_key)
    content_validator = ContentValidator()
    health_predictor = HealthPredictor()
    
    tenant_tag = os.getenv("TENANT_TAG", "modal-default")
    node_freq = os.getenv("ANYNODE_FREQ", "13")
    
    class DiscoveryService:
        def __init__(self):
            self.node_id = f"node-{uuid.uuid4()}"
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
                latency = ping3.ping("8.8.8.8", unit="ms") or 1000  # More reliable target
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

        async def register_node(self, host: str, port: int, freq: str, tenant: str):
            try:
                cache_key = f"{self.node_id}:{freq}:{tenant}"
                if cache_key in self.health_cache and (datetime.now().timestamp() - self.health_cache[cache_key]["timestamp"]) < 13:
                    health = self.health_cache[cache_key]["health"]
                else:
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    latency = ping3.ping("8.8.8.8", unit="ms") or 1000
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
                    "tenant": tenant,
                    "freq": freq,
                    "health": health
                }
                pulse_json = json.dumps(pulse_data)
                
                # Basic content validation instead of ClamAV
                if not content_validator.validate_content(pulse_json):
                    print(f"⚠️ Suspicious content detected in pulse")
                    return
                
                encrypted_pulse, nonce = cipher.encrypt(pulse_json)
                pulse_signature = hmac.new(hmac_key, pulse_json.encode(), hashlib.sha256).hexdigest()
                
                payload = json.dumps({
                    "encrypted": encrypted_pulse, 
                    "signature": pulse_signature,
                    "nonce": nonce  # Include nonce for decryption
                })
                compressed = zlib.compress(payload.encode('utf-8'))
                
                # Use context manager for socket
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as broadcast_socket:
                    broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    broadcast_socket.sendto(compressed, ('255.255.255.255', 7777))  # Proper broadcast
                
                self.pulse_buffer.append(pulse_data)
                print(f"✅ Node {self.node_id} pulsed (tenant: {tenant}, freq: {freq}, health: {health['health_score']:.2f})")
                
            except Exception as e:
                print(f"❌ Pulse registration failed: {e}")

        async def discover_nodes(self, tenant_filter: str = None, freq_filter: str = None) -> List[Dict]:
            # Simulate discovered nodes for demo
            simulated_nodes = []
            for i in range(3):  # Simulate 3 nodes
                health_score = 0.7 + (i * 0.1)  # Varying health scores
                simulated_nodes.append({
                    "id": f"simulated-node-{i}",
                    "address": f"10.0.1.{i+1}",
                    "port": 7777,
                    "tags": [f"tenant:{tenant_tag}", f"freq:{node_freq}"],
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "nexus_address": {"region": 1, "node_type": 1, "role_id": 1, "unit_id": i+1},
                    "health": {
                        "cpu_usage": 20.0 + (i * 5),
                        "memory_usage": 30.0 + (i * 5),
                        "latency_ms": 50.0 + (i * 10),
                        "disk_io_ops": 100 * (i + 1),
                        "success_rate": 0.95,
                        "health_score": health_score
                    }
                })
            
            return sorted(simulated_nodes, key=lambda x: x["health"]["health_score"], reverse=True)

        async def shutdown(self):
            self._stop_health_monitor.set()

    discovery_service = DiscoveryService()
    
    # Start health monitoring
    health_task = asyncio.create_task(discovery_service.health_monitor())
    
    await discovery_service.register_node(
        host="metatron-router.modal.run",
        port=443,
        freq=node_freq,
        tenant=tenant_tag
    )
    
    nodes = await discovery_service.discover_nodes(tenant_filter=None, freq_filter=node_freq)
    node_count = len(nodes)
    print(f"Routing {media_type} across {node_count} nodes (freq: {node_freq})")
    
    # Content validation for input data
    input_data = json.dumps({"grid_size": size, "query_load": query_load})
    if not content_validator.validate_content(input_data):
        raise ValueError("Suspicious content detected in input parameters")
    
    # Route using quantum or fallback method
    if use_quantum and len(nodes) > 1:
        assignments = quantum_walk_route(nodes, query_load, media_type)
        routing_mode = "quantum"
    else:
        grid_data = consciousness_grid(size)
        assignments = route_with_prime_weights(grid_data[0], query_load)
        
        # Enhance assignments with node information
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
        
        routing_mode = "ulam"
    
    # Cleanup
    await discovery_service.shutdown()
    health_task.cancel()
    
    return {
        "grid": consciousness_grid(size)[0].tolist() if not use_quantum else [],
        "assignments": assignments,
        "signal_map": consciousness_grid(size)[1] if not use_quantum else {},
        "discovered_nodes": nodes,
        "media_type": media_type,
        "routing_mode": routing_mode,
        "timestamp": datetime.now().isoformat()
    }