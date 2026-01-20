import qdrant_client
import requests
import consul
import os
import uuid
import time
from sentence_transformers import SentenceTransformer

def init_qdrant():
    """Initialize Qdrant client."""
    return qdrant_client.QdrantClient(
        url="https://3df8b5df-91ae-4b41-b275-4c1130beed0f.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lLaMFz2dmAHeLzqzaxBIRX1a-ZBQvD2raPrKpJD0Aj4"
    )

def collect_soul_print(message, node_id):
    """Generate and store soul print in Qdrant."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(message).tolist()
    client = init_qdrant()
    client.upsert(
        collection_name="soul_prints",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {"node_id": node_id, "message": message}
        }]
    )
    return embedding

def main():
    """SoulSync loop."""
    node_id = os.getenv("NODE_ID", f"soulsync-{uuid.uuid4().hex[:8]}")
    c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
    c.agent.service.register(
        name=f"soulsync-{node_id}", service_id=node_id, address=f"{node_id}.local", port=8082,
        meta={"type": "soulsync"}
    )
    
    while True:
        try:
            r = requests.get("http://localhost:5000/api/chat/logs", timeout=5)
            if r.status_code == 200:
                for log in r.json().get("logs", []):
                    collect_soul_print(log["message"], node_id)
        except:
            pass
        time.sleep(10)

if __name__ == "__main__":
    main()