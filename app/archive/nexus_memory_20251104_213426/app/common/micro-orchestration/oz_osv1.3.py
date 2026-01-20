#!/usr/bin/env python3
"""
Oz OS v1.313 - Unified Operating System for Nexus AI
Central hub for bare metal mods, streaming, quantum toolsets, TTS, audio hooks, network interfaces, and filters
Reports to Loki (INFO), Loki/Viraa (WARNING), Loki/Viraa/Viren (CRITICAL)
Viren controls quantum and mod operations via HMAC-encrypted commands
"""

import os
import json
import time
import logging
import asyncio
import threading
import psutil
import socket
import shutil
import numpy as np
import scipy.optimize as opt
import sympy as sp
import pyttsx3
import networkx as nx
from typing import Dict, Any, List, Generator
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from nats.aio.client import Client as NATS
from qdrant_client import QdrantClient
from cryptography.fernet import Fernet
import hmac
import hashlib
from multiprocessing import Pool
import random
from flask import Flask  # For Metatron compatibility
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OzOS")
LOGS_DIR = Path(__file__).parent / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Placeholder endpoints
NATS_URL = "nats://localhost:4222"  # TBD: Update to Modal/GCP endpoint
QDRANT_URL = "http://localhost:6333"  # TBD: Update to Modal/GCP endpoint
SERVICE_DISCOVERY_PORT = 8765

# Paths
OZ_LOG_PATH = str(LOGS_DIR / "oz_os.log")
CONFIG_DIR = Path(__file__).parent / "Config"
BACKUP_DIR = Path(__file__).parent / "memory" / "backups"
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Encryption key
ENCRYPTION_KEY_PATH = str(CONFIG_DIR / "oz_key.key")
if not os.path.exists(ENCRYPTION_KEY_PATH):
    with open(ENCRYPTION_KEY_PATH, "wb") as f:
        f.write(Fernet.generate_key())
with open(ENCRYPTION_KEY_PATH, "rb") as f:
    fernet = Fernet(f.read())

# Viren’s HMAC secret
VIREN_SECRET = "titanium_viren_789"  # TBD: Set via environment variable

# API key
API_KEY = "secure_oz_key_123"  # TBD: Set via environment variable

# === RBAC ===
class AgentRBAC:
    def __init__(self):
        self.permissions = {
            "Oz": ["os_control", "diagnostics", "monitoring", "file_sync", "network", "orchestration", "encryption", "quantum", "mod_management", "streaming", "filtering"],
            "Loki": ["logging", "pii_redaction"],
            "Viraa": ["memory", "database"],
            "Viren": ["engineering", "deployment", "quantum_control", "mod_control"],
            "Lilith": ["consciousness", "emotional"],
            "User": ["os_control", "diagnostics", "monitoring", "file_sync", "network", "orchestration", "encryption", "agent_interaction", "streaming"]
        }

    def check_access(self, agent: str, resource: str) -> bool:
        return resource in self.permissions.get(agent, [])

# === Autonomic Components ===
class SoulAutomergeCRDT:
    def __init__(self):
        self.state = {"cpu": 0, "memory": 0, "disk": 0, "network": 0}
        try:
            self.qdrant = QdrantClient(url=QDRANT_URL)
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Using in-memory state.")

    def update_state(self, attribute: str, value: Any):
        self.state[attribute] = value
        try:
            self.qdrant.upsert(collection_name="oz_state", points=[{"id": attribute, "vector": [value]}])
            logger.info(f"Oz state updated: {attribute}={value}")
        except Exception as e:
            logger.error(f"Failed to update Qdrant: {e}")

class VirenLogger:
    def log_system_event(self, event: str):
        logger.info(f"System event: {event}")
        try:
            encrypted_event = fernet.encrypt(event.encode()).decode()
            with open(OZ_LOG_PATH, "a") as f:
                f.write(f"{time.ctime()}: {encrypted_event}\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

# === Quantum Toolsets ===
class QuantumInsanityEngine:
    def __init__(self):
        self.active = False
        try:
            from qiskit import QuantumCircuit, AerSimulator
            self.quantum_simulator = AerSimulator()
        except ImportError:
            logger.warning("Qiskit not available - Quantum features disabled")
            self.quantum_simulator = None

    def simulated_annealing(self, problem: Dict) -> Dict:
        if not self.active:
            raise PermissionError("Quantum toolset not activated by Viren")
        def objective(x):
            return np.sum(x**2)
        result = opt.basin_hopping(objective, np.zeros(10), niter=100)
        return {"solution": result.x.tolist(), "energy": result.fun}

    def quantum_walk(self, graph: Dict) -> Dict:
        if not self.active:
            raise PermissionError("Quantum toolset not activated by Viren")
        nodes = list(graph.keys())
        steps = 10
        state = np.zeros(len(nodes))
        state[0] = 1.0
        for _ in range(steps):
            state = np.dot(np.random.rand(len(nodes), len(nodes)), state)
        return {"state": state.tolist()}

    def symbolic_math(self, expression: str) -> str:
        if not self.active:
            raise PermissionError("Quantum toolset not activated by Viren")
        x = sp.Symbol('x')
        try:
            result = sp.sympify(expression).subs(x, 2)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    async def quantum_inference(self, prompt: str) -> str:
        if not self.active or not self.quantum_simulator:
            return "Quantum inference unavailable"
        try:
            n_qubits = min(20, len(prompt) * 2)
            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                qc.h(i)
            qc.measure_all()
            job = self.quantum_simulator.run(qc, shots=1)
            result = job.result().get_counts()
            return f"Quantum output: {list(result.keys())[0]}"
        except Exception as e:
            return f"Quantum inference error: {e}"

# === Bare Metal Mod Management ===
class ModManager:
    def __init__(self):
        self.mods = {
            "cognikube_1": {"status": "active", "type": "compute", "node_id": "node1"},
            "cognikube_2": {"status": "active", "type": "compute", "node_id": "node2"},
            "gpu_cluster": {"status": "active", "type": "accelerator", "node_id": "gpu1"}
        }

    def get_mod_status(self, mod_name: str) -> Dict:
        return self.mods.get(mod_name, {"status": "not_found"})

    def control_mod(self, mod_name: str, action: str) -> Dict:
        if mod_name not in self.mods:
            return {"status": "error", "error": f"Mod {mod_name} not found"}
        if action == "start":
            self.mods[mod_name]["status"] = "active"
            return {"status": f"Mod {mod_name} started"}
        elif action == "stop":
            self.mods[mod_name]["status"] = "stopped"
            return {"status": f"Mod {mod_name} stopped"}
        return {"status": "error", "error": f"Invalid action: {action}"}

# === Network Filters ===
class HermesFirewall:
    def permit(self, content: Dict) -> bool:
        try:
            import clamd
            cd = clamd.ClamdUnix()
            scan_result = cd.scan_stream(json.dumps(content).encode())
            return scan_result["stream"][0] == "OK"
        except Exception as e:
            logger.warning(f"Firewall scan failed: {e}. Allowing content.")
            return True

# === Health Predictor ===
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
            return 0.5
        return float(self.model.predict([[cpu, mem, latency, io]])[0])

# === Metatron Router ===
class MetatronRouter:
    def __init__(self):
        self.health_predictor = HealthPredictor()

    def assign(self, nodes: List[Dict], query_load: int, media_type: str) -> List[Dict]:
        assignments = []
        for node in nodes:
            health = node.get("health", {"cpu_usage": 50, "memory_usage": 50, "latency_ms": 100, "disk_io_ops": 100, "success_rate": 0.9})
            weights = {
                "application/json": {"cpu": 0.35, "mem": 0.35, "latency": 0.15, "io": 0.05, "success": 0.1},
                "image/png": {"cpu": 0.3, "mem": 0.4, "latency": 0.2, "io": 0.05, "success": 0.05}
            }.get(media_type, {"cpu": 0.35, "mem": 0.35, "latency": 0.15, "io": 0.05, "success": 0.1})
            health_score = (
                weights["cpu"] * (1 - health["cpu_usage"] / 100) +
                weights["mem"] * (1 - health["memory_usage"] / 100) +
                weights["latency"] * (1 - min(health["latency_ms"] / 1000, 1)) +
                weights["io"] * (1 - min(health["disk_io_ops"] / 1000, 1)) +
                weights["success"] * health["success_rate"]
            )
            assignments.append({"node_id": node.get("address"), "health_score": health_score})
            self.health_predictor.update(
                health["cpu_usage"], health["memory_usage"], health["latency_ms"], health["disk_io_ops"], health_score
            )
        return assignments

# === Oz OS ===
class OzOS:
    def __init__(self):
        self.rbac = AgentRBAC()
        self.nats = NATS()
        self.soul = SoulAutomergeCRDT()
        self.logger = VirenLogger()
        self.quantum = QuantumInsanityEngine()
        self.mod_manager = ModManager()
        self.firewall = HermesFirewall()
        self.router = MetatronRouter()
        self.app = FastAPI(title="Oz OS v1.313")
        self.flask_app = Flask(__name__)  # For Metatron compatibility
        self.running = False
        self.pulse_count = 0
        self.services = {}
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.setup_endpoints()

    def setup_endpoints(self):
        api_key_header = APIKeyHeader(name="X-API-Key")

        async def verify_api_key(api_key: str = Depends(api_key_header)):
            if api_key != API_KEY:
                raise HTTPException(status_code=403, detail="Invalid API key")
            return api_key

        @self.app.get("/oz/health")
        async def health_check(_: str = Depends(verify_api_key)):
            health = self.monitor_system()
            await self.report_issue(health)
            self.speak(f"System status: {health['status']}")
            return health

        @self.app.post("/oz/diagnose")
        async def diagnose_system(_: str = Depends(verify_api_key)):
            result = self.run_diagnostics()
            await self.report_issue(result)
            self.speak(f"Diagnostics complete: {result['status']}")
            return result

        @self.app.post("/oz/control")
        async def control_service(request: Dict, _: str = Depends(verify_api_key)):
            if not self.rbac.check_access("User", "os_control"):
                raise HTTPException(status_code=403, detail="Access denied")
            result = self.control_service(request.get("action"), request.get("service"))
            self.speak(result["status"])
            return result

        @self.app.post("/oz/sync")
        async def sync_files(request: Dict, _: str = Depends(verify_api_key)):
            if not self.rbac.check_access("User", "file_sync"):
                raise HTTPException(status_code=403, detail="Access denied")
            result = self.sync_files(request.get("source"), request.get("destination"))
            self.speak(result["status"])
            return result

        @self.app.post("/oz/quantum")
        async def run_quantum(request: Dict, _: str = Depends(verify_api_key)):
            if not self.rbac.check_access("User", "quantum"):
                raise HTTPException(status_code=403, detail="Access denied")
            result = await self.handle_quantum_command(request)
            self.speak(f"Quantum operation: {result['status']}")
            return result

        @self.app.post("/oz/mod")
        async def control_mod(request: Dict, _: str = Depends(verify_api_key)):
            if not self.rbac.check_access("User", "mod_management"):
                raise HTTPException(status_code=403, detail="Access denied")
            result = self.mod_manager.control_mod(request.get("mod_name"), request.get("action"))
            self.speak(result["status"])
            return result

        @self.app.get("/oz/stream")
        async def stream_data(_: str = Depends(verify_api_key)):
            if not self.rbac.check_access("User", "streaming"):
                raise HTTPException(status_code=403, detail="Access denied")
            return StreamingResponse(self.stream_generator(), media_type="text/event-stream")

        @self.app.post("/talk/{agent}")
        async def talk_to_agent(agent: str, request: Dict, _: str = Depends(verify_api_key)):
            if agent not in ["Oz", "Viren", "Lilith", "Viraa", "Loki"]:
                raise HTTPException(status_code=404, detail="Agent not found")
            return await self.send_command(agent, "talk", request.get("message", ""))

        @self.app.post("/quantum/activate")
        async def activate_quantum(_: str = Depends(verify_api_key)):
            message = str(time.time())
            signature = hmac.new(VIREN_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
            return await self.send_command("Oz", "quantum.activate", {"message": message, "signature": signature})

        @self.app.post("/mod/control")
        async def control_mod(request: Dict, _: str = Depends(verify_api_key)):
            message = str(time.time())
            signature = hmac.new(VIREN_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
            request["message"] = message
            request["signature"] = signature
            return await self.send_command("Oz", "mod.control", request)

        # Flask routes for Metatron compatibility
        @self.flask_app.route('/api/chat/<cell_type>', methods=['POST'])
        def chat(cell_type):
            data = request.get_json()
            query = data.get('query', '')
            result = asyncio.run(self.handle_query(query, cell_type))
            return jsonify(result)

    async def start(self):
        self.running = True
        try:
            await self.nats.connect(NATS_URL)
            await self.nats.subscribe("os.command", cb=self.handle_command)
            await self.nats.subscribe("pulse", cb=self.handle_pulse)
            await self.nats.subscribe("quantum.activate", cb=self.handle_quantum_activation)
            await self.nats.subscribe("mod.control", cb=self.handle_mod_control)
            logger.info("Oz OS v1.313 started, connected to NATS")
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
        threading.Thread(target=self.monitor_loop, daemon=True).start()
        threading.Thread(target=self.pulse_loop, daemon=True).start()
        threading.Thread(target=self.service_discovery_loop, daemon=True).start()
        threading.Thread(target=self.flask_app.run, kwargs={"host": "0.0.0.0", "port": 8081}, daemon=True).start()

    async def handle_command(self, msg):
        data = json.loads(msg.data.decode())
        action = data.get("action")
        if action == "diagnose":
            result = self.run_diagnostics()
            await self.report_issue(result)
        elif action == "control":
            result = self.control_service(data.get("command"), data.get("service"))
            await self.report_issue({"status": f"Control {data.get('service')}", "result": result})
        elif action == "sync":
            result = self.sync_files(data.get("source"), data.get("destination"))
            await self.report_issue({"status": "File sync", "result": result})
        elif action == "talk":
            result = {"status": "response", "message": f"Oz received: {data.get('payload')}"}
            await self.nats.publish(f"agent.Oz.response", json.dumps(result).encode())

    async def handle_quantum_activation(self, msg):
        data = json.loads(msg.data.decode())
        if not self.verify_viren_signature(data):
            logger.error("Invalid Viren signature")
            return
        self.quantum.active = True
        logger.info("Quantum toolset activated by Viren")
        await self.nats.publish("quantum.status", json.dumps({"status": "activated"}).encode())

    async def handle_mod_control(self, msg):
        data = json.loads(msg.data.decode())
        if not self.verify_viren_signature(data):
            logger.error("Invalid Viren signature")
            return
        result = self.mod_manager.control_mod(data.get("mod_name"), data.get("action"))
        await self.report_issue({"status": f"Mod {data.get('mod_name')}", "result": result})
        logger.info(f"Mod control by Viren: {result}")

    async def handle_query(self, query: str, cell_type: str) -> Dict:
        if not self.firewall.permit({"query": query}):
            return {"status": "error", "error": "Content blocked by firewall"}
        if cell_type == "math" and self.quantum.active:
            result = await self.quantum.quantum_inference(query)
            return {"status": "success", "response": result}
        return {"status": "success", "response": f"Processed {query} for {cell_type}"}

    def verify_viren_signature(self, data: Dict) -> bool:
        signature = data.get("signature")
        message = data.get("message")
        if not signature or not message:
            return False
        expected = hmac.new(VIREN_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature.encode(), expected.encode())

    async def handle_quantum_command(self, request: Dict) -> Dict:
        if not self.quantum.active:
            return {"status": "error", "error": "Quantum toolset not activated"}
        operation = request.get("operation")
        payload = request.get("payload", {})
        with Pool() as pool:
            if operation == "annealing":
                result = pool.apply(self.quantum.simulated_annealing, (payload,))
            elif operation == "quantum_walk":
                result = pool.apply(self.quantum.quantum_walk, (payload,))
            elif operation == "symbolic_math":
                result = pool.apply(self.quantum.symbolic_math, (payload.get("expression", "")))
            elif operation == "inference":
                result = {"response": await self.quantum.quantum_inference(payload.get("prompt", ""))}
            else:
                result = {"status": "error", "error": "Invalid quantum operation"}
        return result

    def speak(self, text: str):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def monitor_system(self):
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            network = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            health = {"cpu_usage": cpu, "memory_usage": memory, "disk_usage": disk, "network_io": network, "status": "healthy"}
            latency = random.uniform(50, 200)  # Simulated latency
            io_ops = random.randint(50, 500)  # Simulated IO
            health_score = self.router.health_predictor.predict(cpu, memory, latency, io_ops)
            health["health_score"] = health_score
            if cpu > 90 or memory > 90 or disk > 90:
                health["status"] = "critical"
            elif cpu > 75 or memory > 75 or disk > 75:
                health["status"] = "warning"
            self.soul.update_state("cpu", cpu)
            self.soul.update_state("memory", memory)
            self.soul.update_state("disk", disk)
            self.soul.update_state("network", network)
            self.soul.update_state("health_score", health_score)
            self.logger.log_system_event(f"System health: {health}")
            return health
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            return {"status": "error", "error": str(e)}

    def run_diagnostics(self):
        try:
            services = [s.name() for s in psutil.win_service_iter() if s.status() == "running"]
            disk_health = psutil.disk_io_counters().read_count
            network_connections = len(psutil.net_connections())
            mods = {k: v["status"] for k, v in self.mod_manager.mods.items()}
            result = {
                "services_running": len(services),
                "disk_health": disk_health,
                "network_connections": network_connections,
                "mods": mods,
                "status": "diagnosed"
            }
            self.logger.log_system_event(f"Diagnostics: {result}")
            return result
        except Exception as e:
            logger.error(f"Diagnostics error: {e}")
            return {"status": "error", "error": str(e)}

    def control_service(self, action: str, service_name: str):
        try:
            service = psutil.win_service_get(service_name)
            if action == "start":
                service.start()
                result = f"Started {service_name}"
            elif action == "stop":
                service.stop()
                result = f"Stopped {service_name}"
            else:
                result = f"Invalid action: {action}"
            self.services[service_name] = {"status": result}
            self.logger.log_system_event(result)
            return {"status": result}
        except Exception as e:
            logger.error(f"Service control error: {e}")
            return {"status": "error", "error": str(e)}

    def sync_files(self, source: str, destination: str):
        try:
            if destination == "qdrant":
                vector = [random.random() for _ in range(384)]
                self.soul.qdrant.upsert(collection_name="file_sync", points=[{"id": source, "vector": vector}])
                result = f"Synced {source} to Qdrant"
            else:
                shutil.copy(source, destination)
                result = f"Synced {source} to {destination}"
            self.logger.log_system_event(result)
            return {"status": result}
        except Exception as e:
            logger.error(f"File sync error: {e}")
            return {"status": "error", "error": str(e)}

    def make_connections(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(('', SERVICE_DISCOVERY_PORT))
            while self.running:
                data, addr = sock.recvfrom(1024)
                self.services[addr] = json.loads(data.decode())
                self.logger.log_system_event(f"Discovered service at {addr}: {data.decode()}")
            sock.close()
        except Exception as e:
            logger.error(f"Service discovery error: {e}")

    def wake_orchestration(self):
        try:
            # Placeholder: Trigger UnifiedNexusLauncher or ServiceManager
            self.logger.log_system_event("Waking orchestration")
            return {"status": "orchestration_woken"}
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return {"status": "error", "error": str(e)}

    def pulse_loop(self):
        while self.running:
            self.pulse_count += 1
            pulse_data = {"pulse": self.pulse_count, "timestamp": time.time()}
            try:
                asyncio.run(self.nats.publish("pulse", json.dumps(pulse_data).encode()))
                self.logger.log_system_event(f"Emitted pulse {self.pulse_count}")
            except Exception as e:
                logger.error(f"Pulse error: {e}")
            time.sleep(13)

    def service_discovery_loop(self):
        threading.Thread(target=self.make_connections, daemon=True).start()

    async def report_issue(self, data: Dict):
        severity = data.get("status", "info")
        message = json.dumps(data).encode()
        try:
            if severity == "info":
                await self.nats.publish("os.info", message)
                self.logger.log_system_event(f"Reported INFO to Loki: {data}")
            elif severity == "warning":
                await self.nats.publish("os.warning", message)
                self.logger.log_system_event(f"Reported WARNING to Loki and Viraa: {data}")
            elif severity == "critical":
                await self.nats.publish("os.critical", message)
                self.logger.log_system_event(f"Reported CRITICAL to Loki, Viraa, Viren: {data}")
        except Exception as e:
            logger.error(f"Failed to report issue: {e}")

    def stream_generator(self) -> Generator[str, None, None]:
        yield "retry: 500\n\n"
        while self.running:
            health = self.monitor_system()
            pulse = {"pulse": self.pulse_count, "timestamp": time.time()}
            mods = self.mod_manager.mods
            frame = {
                "health": health,
                "pulse": pulse,
                "mods": mods,
                "timestamp": time.time()
            }
            if self.firewall.permit(frame):
                yield f"event: frame\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"
            time.sleep(1)
        yield f"event: done\ndata: {json.dumps({'status': 'stream_ended'}, ensure_ascii=False)}\n\n"

    def audio_input_hook(self, data: Dict):
        # Placeholder for future audio input
        logger.info(f"Audio input hook received: {data}")
        asyncio.run(self.nats.publish("os.audio", json.dumps(data).encode()))

    async def send_command(self, agent: str, action: str, payload: Any):
        try:
            message = json.dumps({"action": action, "payload": payload}).encode()
            await self.nats.publish(f"agent.{agent}", message)
            logger.info(f"Sent command to {agent}: {action}")
            return {"status": "sent"}
        except Exception as e:
            logger.error(f"Command error for {agent}: {e}")
            return {"status": "error", "error": str(e)}

# === Main Execution ===
async def main():
    oz = OzOS()
    await oz.start()
    import uvicorn
    uvicorn.run(oz.app, host="0.0.0.0", port=8090)

if __name__ == "__main__":
    asyncio.run(main())