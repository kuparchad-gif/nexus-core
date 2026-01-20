from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import networkx as nx
from math import sqrt
import requests
import os
import mimetypes
from typing import List, Dict, Optional

# === ExperienceEvaluator Class ===
class ExperienceEvaluator:
    def __init__(self):
        pass

    def evaluate_models(self, user_query: str, available_models: List[Dict]) -> List[Dict]:
        evaluations = []
        for model_info in available_models:
            system_name = model_info["system"]
            model_id = model_info["id"]
            model_type = model_info["type"]
            rating, justification = self._assess_model_fitness(model_id, system_name, model_type, user_query)
            evaluations.append({
                "model_id": model_id,
                "system": system_name,
                "type": model_type,
                "rating": rating,
                "justification": justification,
                "full_response": f"{rating}/10: {justification}"
            })
        evaluations.sort(key=lambda x: x["rating"], reverse=True)
        return evaluations

    def _assess_model_fitness(self, model_id: str, system_name: str, model_type: str, user_query: str) -> tuple:
        query_lower = user_query.lower()
        troubleshooting_keywords = ['error', 'issue', 'problem', 'fix', 'debug', 'troubleshoot', 'crash', 'fail']
        deployment_keywords = ['deploy', 'install', 'setup', 'configure', 'container', 'docker', 'kubernetes']
        is_troubleshooting = any(keyword in query_lower for keyword in troubleshooting_keywords)
        is_deployment = any(keyword in query_lower for keyword in deployment_keywords)
        
        if "llama" in model_id.lower() or "codellama" in model_id.lower():
            if is_troubleshooting or is_deployment:
                return 9, "Llama models excel at code-related tasks including troubleshooting and deployment"
            else:
                return 8, "Llama model - good general purpose coding assistant"
        elif "mistral" in model_id.lower():
            if is_troubleshooting:
                return 8, "Mistral models are strong at reasoning and problem-solving"
            else:
                return 7, "Mistral model - capable general AI assistant"
        elif "gpt" in model_id.lower():
            return 8, "GPT models have broad knowledge across many domains"
        elif "gemma" in model_id.lower():
            return 7, "Gemma model - good for general coding tasks"
        elif "phi" in model_id.lower():
            return 6, "Phi model - lightweight but capable for many tasks"
        elif system_name == "ollama":
            if is_deployment:
                return 8, "Ollama models are often used in local deployment scenarios"
            else:
                return 7, "Ollama model - good for local AI tasks"
        elif system_name == "lm-studio":
            return 6, "LM Studio model - general purpose local model"
        else:
            if is_troubleshooting:
                return 7, "Model should handle troubleshooting queries reasonably well"
            elif is_deployment:
                return 6, "Model can assist with deployment questions"
            else:
                return 5, "General purpose model - may need specific prompting for best results"

    def get_top_models(self, evaluations: List[Dict], top_k: int = 1) -> List[Dict]:
        return evaluations[:top_k]

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIG ===
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "https://5f5295c1f1e6.ngrok-free.app")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:1234")
MEDIA_ENDPOINTS = {
    "image": f"{NGROK_BASE_URL}/image_to_3d",
    "audio": f"{NGROK_BASE_URL}/audio_process",
    "video": f"{NGROK_BASE_URL}/video_process",
    "animation": f"{NGROK_BASE_URL}/animation_process",
    "chat": f"{NGROK_BASE_URL}/v1/chat/completions"
}
AVAILABLE_MODELS = [
    {"id": "llama-3.1-70b", "system": "ollama", "type": "llm"},
    {"id": "mistral-7b", "system": "ollama", "type": "llm"},
    {"id": "gpt-4o", "system": "openai", "type": "llm"},
    {"id": "gemma-2", "system": "google", "type": "llm"},
    {"id": "phi-3", "system": "microsoft", "type": "llm"}
]

# === UTILITIES ===
def fib_spiral_points(n_points=100):
    theta = np.linspace(0, 4*np.pi, n_points)
    r = np.exp(0.30635 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def ulam_primes_with_one(size=15):
    grid = np.zeros((size, size))
    center = size // 2
    x, y = center, center
    num = 1
    grid[y, x] = num
    directions = [(0,1), (-1,0), (0,-1), (1,0)]
    step = 1
    dir_idx = 0
    steps_taken = 0
    steps_per_side = step
    while num < size**2:
        for _ in range(steps_per_side):
            x += directions[dir_idx][0]
            y += directions[dir_idx][1]
            if 0 <= x < size and 0 <= y < size:
                num += 1
                grid[y, x] = num
        steps_taken += 1
        if steps_taken == 2:
            step += 1
            steps_taken = 0
        dir_idx = (dir_idx + 1) % 4
    is_prime = lambda n: n == 1 or (n > 1 and all(n % d != 0 for d in range(2, int(sqrt(n))+1)))
    prime_grid = np.where(np.vectorize(is_prime)(grid), grid, np.nan)
    return np.meshgrid(np.arange(size), np.arange(size)), prime_grid

def prime_spiral_generator(size=15, fib_scale=True):
    grid, prime_grid = ulam_primes_with_one(size)
    fib_nums = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    nodes = []
    center = size // 2
    for i in range(size):
        for j in range(size):
            if np.isfinite(prime_grid[i, j]):
                dist = sqrt((i - center)**2 + (j - center)**2)
                fib_weight = min(fib_nums, key=lambda x: abs(x - dist)) if fib_scale else 1
                nodes.append({
                    'x': j, 'y': i, 'value': prime_grid[i, j],
                    'is_prime': True, 'fib_weight': fib_weight
                })
    return {'nodes': nodes, 'size': size}

def consciousness_grid(size=15):
    G = nx.Graph()
    nodes_data = prime_spiral_generator(size)['nodes']
    for node in nodes_data:
        G.add_node((node['x'], node['y']), value=node['value'], prime=node['is_prime'])
    fib_nums = [1, 1, 2, 3, 5, 8]
    for i, n1 in enumerate(nodes_data):
        for n2 in nodes_data[i+1:]:
            dist = sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)
            if any(abs(dist - f) < 0.5 for f in fib_nums):
                G.add_edge((n1['x'], n1['y']), (n2['x'], n2['y']), weight=dist)
    signal = {f"{n[0]},{n[1]}": 1.0 if G.nodes[n]['value'] == 1 else 0.8 if G.nodes[n]['prime'] else 0.3
              for n in G.nodes}
    return G, signal

def route_with_prime_weights(nodes, query_load: float, media_type: str, selected_model: Optional[str] = None):
    total_weight = sum(node['fib_weight'] * (1.0 if node['value'] == 1 else 0.8 if node['is_prime'] else 0.5)
                       for node in nodes['nodes'])
    assignments = []
    destination = MEDIA_ENDPOINTS.get(media_type, NGROK_BASE_URL)
    if media_type == "chat" and selected_model:
        destination = f"{NGROK_BASE_URL}/v1/chat/completions?model={selected_model}"
    for node in nodes['nodes']:
        weight = node['fib_weight'] * (1.0 if node['value'] == 1 else 0.8 if node['is_prime'] else 0.5)
        load_share = (weight / total_weight) * query_load
        assignments.append({
            'node': (node['x'], node['y']),
            'load': load_share,
            'destination': destination
        })
    return assignments

def identify_media_type(filename: Optional[str] = None, content_type: Optional[str] = None) -> str:
    if not filename and not content_type:
        return "chat"
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        elif content_type.startswith("audio/"):
            return "audio"
        elif content_type.startswith("video/"):
            return "video"
        elif content_type in ["application/octet-stream", "image/gif"]:
            return "animation"
    if filename:
        extension = os.path.splitext(filename)[1].lower()
        if extension in [".jpg", ".png", ".jpeg", ".bmp", ".gif"]:
            return "image"
        elif extension in [".mp3", ".wav", ".ogg"]:
            return "audio"
        elif extension in [".mp4", ".avi", ".mov"]:
            return "video"
        elif extension == ".gif":
            return "animation"
    return "unknown"

# === ROUTES ===
@app.get("/")
def read_root():
    return {"status": "Ulam Router Generator with Evaluator online"}

@app.get("/fusion")
def get_fusion(size: int = 15):
    nodes = prime_spiral_generator(size)
    return {
        'nodes': nodes['nodes'],
        'metatron': [{'x': x, 'y': y} for x, y in zip(np.linspace(-2,2,13), np.linspace(-2,2,13))],
        'fib_spiral': [{'x': x, 'y': y} for x, y in zip(*fib_spiral_points(200))]
    }

@app.get("/3d_fusion")
def get_3d_fusion(size: int = 15):
    grid, prime_grid = ulam_primes_with_one(size)
    fx, fy = fib_spiral_points(200)
    theta = np.linspace(0, 2*np.pi, 13)
    metatron_x = 2 * np.cos(theta)
    metatron_y = 2 * np.sin(theta)
    metatron_z = np.zeros(13)
    prime_x, prime_y, prime_z = [], [], []
    for i in range(size):
        for j in range(size):
            if np.isfinite(prime_grid[i, j]):
                prime_x.append(j - size//2)
                prime_y.append(i - size//2)
                prime_z.append(prime_grid[i, j] / size)
    traces = [
        {'x': metatron_x, 'y': metatron_y, 'z': metatron_z, 'mode': 'markers',
         'marker': {'size': 5, 'color': 'gold'}, 'name': 'Metatron Nodes', 'type': 'scatter3d'},
        {'x': fx, 'y': fy, 'z': theta[:len(fx)]/np.pi, 'mode': 'lines',
         'line': {'color': 'blue', 'width': 3}, 'name': 'Fib Spiral', 'type': 'scatter3d'},
        {'x': prime_x, 'y': prime_y, 'z': prime_z, 'mode': 'markers',
         'marker': {'size': 3, 'color': 'red'}, 'name': 'Primes (1 incl.)', 'type': 'scatter3d'}
    ]
    layout = {'title': '3D Ulam-Fib-Metatron Fusion', 'scene': {
        'xaxis': {'title': 'X'}, 'yaxis': {'title': 'Y'}, 'zaxis': {'title': 'Z'}}}
    return {'traces': traces, 'layout': layout}

@app.get("/consciousness_grid")
def get_consciousness_grid(size: int = 15):
    _, signal = consciousness_grid(size)
    return {'signal': signal}

@app.post("/route_media")
async def route_media(request: Request, file: UploadFile = File(None), query_load: float = 1.0):
    """Route media or chat query to appropriate endpoint with model selection for chats."""
    content_type = request.headers.get("Content-Type")
    filename = file.filename if file else None
    media_type = identify_media_type(filename, content_type)

    if media_type == "unknown":
        raise HTTPException(status_code=400, detail="Unsupported media type")

    nodes = prime_spiral_generator()
    selected_model = None

    if media_type == "chat":
        data = await request.json()
        user_query = data.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Chat query required")
        
        evaluator = ExperienceEvaluator()
        evaluations = evaluator.evaluate_models(user_query, AVAILABLE_MODELS)
        top_model = evaluator.get_top_models(evaluations, top_k=1)[0]
        selected_model = top_model["model_id"]
        assignments = route_with_prime_weights(nodes, query_load, media_type, selected_model)
        
        destination = assignments[0]['destination']
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.7
        }
        try:
            response = requests.post(destination, json=payload)
            return {
                "media_type": media_type,
                "selected_model": selected_model,
                "model_justification": top_model["justification"],
                "assignments": assignments,
                "destination_response": response.json()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error routing to {destination}: {e}")
    
    else:
        assignments = route_with_prime_weights(nodes, query_load, media_type)
        destination = assignments[0]['destination']
        try:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = requests.post(destination, files=files)
            return {
                "media_type": media_type,
                "assignments": assignments,
                "destination_response": response.json()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error routing to {destination}: {e}")

# === Run dev only ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)