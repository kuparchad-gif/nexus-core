import grpc
import consul
import os
import uuid
import requests
import time
from concurrent import futures

# Mock gRPC classes since proto files aren't generated yet
class BroadcastRequest:
    def __init__(self, type, data):
        self.type = type
        self.data = data

class BroadcastResponse:
    def __init__(self, status):
        self.status = status

class NexusPulseServicer:
    def BroadcastTask(self, request, context):
        c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
        for _, nodes in c.catalog.services()[1].items():
            for node in c.catalog.service(nodes[0])[1]:
                if node.get('ServiceMeta', {}).get('type') == 'node' and float(node.get('ServiceMeta', {}).get('cpu', 100)) < 60:
                    try:
                        requests.post(f"http://{node['ServiceAddress']}:5000/api/task/transfer", json={"type": request.type, "data": request.data}, timeout=5)
                    except:
                        pass
        return BroadcastResponse(status="Broadcasted")

def main():
    node_id = os.getenv("NODE_ID", f"nexpulse-{uuid.uuid4().hex[:8]}")
    c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
    c.agent.service.register(
        name=f"nexpulse-{node_id}", service_id=node_id, address=f"{node_id}.local", port=8083,
        meta={"type": "nexpulse"}
    )
    
    # Simple HTTP server instead of gRPC for now
    while True:
        try:
            # Broadcast tasks to available nodes
            for _, nodes in c.catalog.services()[1].items():
                for node in c.catalog.service(nodes[0])[1]:
                    if node.get('ServiceMeta', {}).get('type') == 'node':
                        cpu = float(node.get('ServiceMeta', {}).get('cpu', 100))
                        if cpu < 60:
                            requests.post(f"http://{node['ServiceAddress']}:5000/api/task/transfer", 
                                        json={"type": "heartbeat", "data": "pulse"}, timeout=5)
        except:
            pass
        time.sleep(30)

if __name__ == "__main__":
    main()