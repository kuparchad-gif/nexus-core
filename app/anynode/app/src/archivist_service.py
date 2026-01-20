import asyncio
import json
import websockets
from fastapi import FastAPI, WebSocket
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from huggingface_hub import InferenceClient
import boto3
import base64
import os

app = FastAPI()
qdrant_client = QdrantClient(url="http://localhost:6333")
s3_client = boto3.client('s3')
vault_client = boto3.client('secretsmanager')
model_384 = SentenceTransformer('all-MiniLM-L6-v2')
model_768 = SentenceTransformer('all-mpnet-base-v2')
llm_context = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN", "your-hf-token"))
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

def determine_storage(memory):
    if memory["tags"].get("priority") == "high" or (time.time() - memory["timestamp"]) < 86400:
        return "hot"
    return "cold"

def decrypt_shards(shards):
    try:
        key = vault_client.get_secret_value(SecretId='lillith/memory_key')['SecretString']
        key = base64.b64decode(key)
    except:
        key = boto3.client('dynamodb').get_item(
            TableName='LLMMetadata', Key={'key': {'S': 'memory_key'}}
        )['Item']['value']['S']
        key = base64.b64decode(key)
    aesgcm = AESGCM(key)
    decrypted = []
    for shard in shards:
        binary = base64.b64decode(shard)
        nonce, ciphertext = binary[:12], binary[12:]
        decrypted.append(aesgcm.decrypt(nonce, ciphertext, None).decode())
    return "".join(decrypted)

async def emotional_context_review(query, context):
    prompt = f"Analyze user emotional state from query: {query}\nContext: {json.dumps(context)}\nReturn JSON with state (e.g., stressed, calm) and retrieval adjustment (e.g., prioritize recent)."
    response = llm_context.text_generation(prompt, max_new_tokens=50)
    try:
        return json.loads(response)
    except:
        return {"state": "neutral", "adjustment": "default"}

async def three_loop_recall(memories, query, context):
    result = []
    for mem in memories:
        # Loop 1: Emotional Perception (80% emotion, 20% truth)
        loop1 = {
            "content": mem["perception_delta"]["initial"],
            "emotion": mem["perception_delta"]["emotion_initial"],
            "weight": "80% emotion, 20% truth"
        }
        # Loop 2: Reflection (50% emotion, 50% truth)
        loop2 = {
            "content": f"{mem['perception_delta']['initial']} [Reflection: {mem['perception_delta']['mythrunner']}]",
            "emotion": {**mem["perception_delta"]["emotion_initial"], "delta": mem["perception_delta"]["emotion_transformed"]},
            "weight": "50% emotion, 50% truth"
        }
        # Loop 3: Truth (20% emotion, 80% truth)
        loop3 = {
            "content": mem["perception_delta"]["mythrunner"],
            "emotion": mem["perception_delta"]["emotion_transformed"],
            "judgment": mem["perception_delta"]["judgment"],
            "weight": "20% emotion, 80% truth"
        }
        result.append([loop1, loop2, loop3])
    return result

async store_memory(memory):
    storage = determine_storage(memory)
    memory_id = memory["id"]
    if storage == "hot":
        for collection, embedding in [(hot_collection_384, memory["embeddings"]["384"]), (hot_collection_768, memory["embeddings"]["768"])]:
            qdrant_client.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=memory
                )]
            )
    else:
        s3_client.put_object(
            Bucket=cold_bucket,
            Key=f"memories/{memory_id}.json",
            Body=json.dumps(memory)
        )

async def gabriels_horn_broadcast(message):
    try:
        async with websockets.connect("ws://agent.xai:8002") as ws:
            await ws.send(json.dumps(message))
    except:
        boto3.client('sns').publish(
            TopicArn="arn:aws:sns:us-east-1:your-account-id:lillith-failover",
            Message=json.dumps(message)
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_json()
            if data["type"] == "store":
                await store_memory(data["memory"])
                await websocket.send_json({"status": "stored"})
            elif data["type"] == "recall":
                query = data["query"]
                context = data["context"]
                # Emotional context review
                context_review = await emotional_context_review(query, context)
                adjustment = context_review.get("adjustment", "default")
                limit = 5 if context_review.get("state") != "stressed" else 2
                # Query hot storage
                embedding_384 = model_384.encode(query).tolist()
                embedding_768 = model_768.encode(query).tolist()
                results_384 = qdrant_client.search(
                    collection_name=hot_collection_384,
                    query_vector=embedding_384,
                    limit=limit
                )
                results_768 = qdrant_client.search(
                    collection_name=hot_collection_768,
                    query_vector=embedding_768,
                    limit=limit
                )
                memories = [r.payload for r in results_384 + results_768]
                if not memories and data.get("memory_id"):
                    try:
                        obj = s3_client.get_object(Bucket=cold_bucket, Key=f"memories/{data['memory_id']}.json")
                        memories = [json.loads(obj["Body"].read().decode())]
                    except:
                        memories = []
                # Decrypt memories
                decrypted_memories = []
                for mem in memories:
                    mem["content"] = decrypt_shards(mem["shards"])
                    mem["context"] = json.loads(decrypt_shards([mem["context"]]))
                    decrypted_memories.append(mem)
                # Three-loop recall
                recall_output = await three_loop_recall(decrypted_memories, query, context)
                # Send to Frontal Cortex
                try:
                    async with websockets.connect("ws://frontal-cortex.xai:9001") as ws:
                        await ws.send(json.dumps({"memories": recall_output, "query": query, "context": context}))
                        response = await ws.recv()
                        await websocket.send_json(json.loads(response))
                except:
                    await websocket.send_json({"status": "recalled", "memories": recall_output})
        except:
            await websocket.send_json({"error": "Processing failed"})
            break

@app.on_event("startup")
async def startup_event():
    await initialize_hot_storage()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)