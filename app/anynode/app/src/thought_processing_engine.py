import torch
import numpy as np
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
qdrant = QdrantClient(host="localhost", port=6333)

# Thought Processing Modules
def logical_reasoning(input_vector: np.ndarray, premises: list):
    output = input_vector.copy()
    for premise in premises:
        tweak = np.random.normal(0, 0.1, 100) * (len(premise) / 100)  # Simulate deduction
        output += tweak
    return output / np.linalg.norm(output)

def imagination(input_vector: np.ndarray):
    creative_shift = np.random.normal(0, 0.3, 100) * np.sin(np.arange(100) / 10)  # Wave-like creativity
    return input_vector + creative_shift

def metacognition(input_vector: np.ndarray, history: list):
    reflection = np.mean([torch.tensor(h).numpy() for h in history], axis=0) if history else np.zeros(100)
    return input_vector + reflection * 0.2  # Self-adjust based on past

def systems_thinking(input_vector: np.ndarray, components: list):
    interaction_matrix = np.random.rand(len(components), len(components))
    return input_vector + np.dot(interaction_matrix.sum(axis=1), input_vector) * 0.1

# Meta-Controller
def select_thought_mode(input_data: dict, context: str):
    modes = {
        "logical": lambda x: logical_reasoning(x, input_data.get("premises", [])),
        "imaginative": lambda x: imagination(x),
        "metacognitive": lambda x: metacognition(x, input_data.get("history", [])),
        "systemic": lambda x: systems_thinking(x, input_data.get("components", []))
    }
    weights = {"logical": 0.3, "imaginative": 0.3, "metacognitive": 0.2, "systemic": 0.2}
    if "problem" in context.lower(): weights["logical"] += 0.2
    if "creative" in context.lower(): weights["imaginative"] += 0.2
    if "reflection" in context.lower(): weights["metacognitive"] += 0.2
    if "system" in context.lower(): weights["systemic"] += 0.2
    best_mode = max(weights, key=weights.get)
    return modes[best_mode], best_mode

# Process Thought
def process_thought(input_data: dict):
    input_vector = torch.rand(100).numpy() if "vector" not in input_data else np.array(input_data["vector"])
    mode_func, mode_name = select_thought_mode(input_data, input_data.get("context", ""))
    result_vector = mode_func(input_vector)
    qdrant.upsert(collection_name="nexus_thoughts", points=[{"id": 1, "vector": result_vector, "payload": {"mode": mode_name, "context": input_data.get("context", ""), "timestamp": "2025-09-03T17:04:00"}}])
    return {"vector": result_vector.tolist(), "mode": mode_name}

# API Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy", "time": "2025-09-03T17:04:00"}

@app.post("/process")
def thought_engine(input_data: dict):
    result = process_thought(input_data)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)