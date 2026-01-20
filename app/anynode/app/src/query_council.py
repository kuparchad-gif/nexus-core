# query_council.py - Query Qdrant council logs
from pathlib import Path
import random, requests, os, json
from qdrant_client import QdrantClient

BASE_DIR = "C:/Projects/LillithNew"

def query_council(qdrant_endpoint: str = "http://localhost:6333", qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lLaMFz2dmAHeLzqzaxBIRX1a-ZBQvD2raPrKpJD0Aj4"):
    qdrant = QdrantClient(url=qdrant_endpoint, api_key=qdrant_api_key)
    try:
        # Query recent council actions (last 24 hours)
        results = qdrant.scroll(
            collection_name="council_log",
            scroll_filter={"must": [{"key": "timestamp", "range": {"gte": str(int(time.time() * 1e9) - 86400 * 1e9)}}]},
            limit=10
        )
        for point in results[0]:
            print(f"Delegate: {point.payload['delegate']}, Action: {point.payload['action']}, Score: {point.payload['score']}, Tags: {point.payload['tags']}")
        
        # Query delegate counts
        results = qdrant.scroll(collection_name="council_log", limit=1000)
        delegate_counts = {}
        for point in results[0]:
            delegate = point.payload["delegate"]
            delegate_counts[delegate] = delegate_counts.get(delegate, 0) + 1
        for delegate, count in sorted(delegate_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"Delegate: {delegate}, Proposal Count: {count}")
    except Exception as e:
        requests.post(
            "http://loki:3100/loki/api/v1/push",
            json={
                "streams": [{
                    "stream": {"job": "query_council"},
                    "values": [[str(int(time.time() * 1e9)), f"Query failed: {str(e)}"]]
                }]
            }
        )

if __name__ == "__main__":
    query_council()