import asyncio
import json
import websockets
import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from huggingface_hub import InferenceClient
import os

app = FastAPI()
qdrant_client = QdrantClient(url="http://localhost:6333")
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')
model_384 = SentenceTransformer('all-MiniLM-L6-v2')
model_768 = SentenceTransformer('all-mpnet-base-v2')
llm_b1 = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN", "your-hf-token"))
llm_b2 = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN", "your-hf-token"))
hot_collection_384 = "hot_memories_384"
hot_collection_768 = "hot_memories_768"
cold_bucket = "lillith-cold-storage"

async def initialize_hot_storage():
    for collection, size in [(hot_collection_384, 384), (hot_collection_768, 768)]:
        try:
            qdrant_client.create_collection(
                collection_name=collection,
                vectors_config={"size": size, "distance": "Cosine"},
                optimizers_config={"memmap_threshold": 20000}
            )
        except:
            print(f"Collection {collection} exists")

async def gabriels_horn_listener():
    try:
        async with websockets.connect("ws://archivist.xai:8765") as ws:
            while True:
                message = await ws.recv()
                data = json.loads(message)
                if data["type"] == "request":
                    await asyncio.gather(
                        handle_request_b1(data),
                        handle_request_b2(data)
                    )
    except:
        sns_client.publish(
            TopicArn="arn:aws:sns:us-east-1:your-account-id:lillith-failover",
            Message=json.dumps(data)
        )

async def handle_request_b1(data):
    query = data["query"]
    context = data["context"]
    prompt = f"Predict memory IDs based on emotional context: {query}\nContext: {json.dumps(context)}\nReturn IDs as comma-separated list."
    prediction = llm_b1.text_generation(prompt, max_new_tokens=50)
    memory_ids = [id.strip() for id in prediction.split(",") if id.strip()]
    await prefetch_memories(memory_ids, hot_collection_384, model_384)

async def handle_request_b2(data):
    query = data["query"]
    context = data["context"]
    prompt = f"Predict memory IDs based on user/session metadata: {query}\nContext: {json.dumps(context)}\nReturn IDs as comma-separated list."
    prediction = llm_b2.text_generation(prompt, max_new_tokens=50)
    memory_ids = [id.strip() for id in prediction.split(",") if id.strip()]
    await prefetch_memories(memory_ids, hot_collection_768, model_768)

async def prefetch_memories(memory_ids, collection, model):
    for memory_id in memory_ids:
        try:
            point = qdrant_client.retrieve(collection_name=collection, ids=[memory_id])
            if not point:
                obj = s3_client.get_object(Bucket=cold_bucket, Key=f"memories/{memory_id}.json")
                memory = json.loads(obj["Body"].read().decode())
                embedding = model.encode(memory["content"]).tolist()
                qdrant_client.upsert(
                    collection_name=collection,
                    points=[PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=memory
                    )]
                )
        except:
            print(f"Failed to prefetch memory {memory_id}")

@app.on_event("startup")
async def startup_event():
    await initialize_hot_storage()
    asyncio.create_task(gabriels_horn_listener())

@app.post("/api/predict")
async def predict_memories(data: dict):
    await asyncio.gather(
        handle_request_b1(data),
        handle_request_b2(data)
    )
    return {"status": "predicted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)