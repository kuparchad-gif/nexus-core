from qdrant_client import QdrantClient
import grpc
from typing import Dict, List
import json
from cryptography.fernet import Fernet
import logging
import asyncio
import websockets
from trumpet_structure import TrumpetStructure

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

class CommunicationLayer:
    def __init__(self, service_name: str):
        self.qdrant = QdrantClient(host='qdrant', port=6333)
        self.logger = setup_logger(f"{service_name}.comm")
        self.cipher = Fernet(Fernet.generate_key())
        self.trumpet = TrumpetStructure()
        self.websocket_servers = {}

    async def send_grpc(self, stub, request, target_pods: List[str]):
        try:
            response = await stub(request)
            self.logger.info({"action": "grpc_send", "targets": target_pods, "status": "success"})
            return response
        except grpc.RpcError as e:
            self.logger.error({"action": "grpc_send", "targets": target_pods, "error": str(e)})
            raise

    def send_http(self, data: Dict, target_pods: List[str], endpoint: str):
        encrypted_data = self.cipher.encrypt(json.dumps(data).encode()).hex()
        for pod in target_pods:
            self.qdrant.upload_collection(
                collection_name="nexus_signals",
                vectors=[[0.1] * 768],
                payload={"data": encrypted_data, "target": pod, "endpoint": endpoint}
            )
        self.logger.info({"action": "http_send", "targets": target_pods, "endpoint": endpoint})

    async def broadcast_to_kubes(self, data: Dict, data_type: str, kubes: Dict):
        tasks = []
        for kube_id, kube in kubes.items():
            if data_type in ['soul', 'signal', 'query']:
                tasks.append(self.send_grpc(None, data, [kube['full_address']]))
                self.send_http(data, [kube['full_address']], f"/{data_type}")
                # WebSocket broadcast
                await self.send_websocket(data, [kube['full_address']])
        await asyncio.gather(*tasks)
        self.logger.info({"action": "broadcast", "data_type": data_type, "kubes": list(kubes.keys())})

    async def start_websocket_server(self, address: str, port: int):
        """Start WebSocket server for a kube."""
        async def handler(websocket, path):
            async for message in websocket:
                decrypted = json.loads(self.cipher.decrypt(bytes.fromhex(message)))
                self.trumpet.pulse_replication({address: self.qdrant})  # Process signal
                self.logger.info({"action": "websocket_receive", "address": address, "data": decrypted})
                await websocket.send(json.dumps({"status": "received"}).encode().hex())
        
        server = await websockets.serve(handler, address, port)
        self.websocket_servers[f"{address}:{port}"] = server
        self.logger.info({"action": "websocket_start", "address": f"{address}:{port}"})

    async def send_websocket(self, data: Dict, target_pods: List[str]):
        """Send data via WebSocket to target pods."""
        for pod in target_pods:
            address, port = pod.split(":")[0], int(pod.split(":")[1])
            try:
                async with websockets.connect(f"ws://{address}:{port}") as ws:
                    encrypted = self.cipher.encrypt(json.dumps(data).encode()).hex()
                    await ws.send(encrypted)
                    response = await ws.recv()
                    self.logger.info({"action": "websocket_send", "target": pod, "response": response})
            except Exception as e:
                self.logger.error({"action": "websocket_send", "target": pod, "error": str(e)})

    def stop_websocket_servers(self):
        """Stop all WebSocket servers."""
        for server in self.websocket_servers.values():
            server.close()
        self.websocket_servers.clear()