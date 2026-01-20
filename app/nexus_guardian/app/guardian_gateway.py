# guardian_gateway.py
import modal
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import networkx as nx
from math import sqrt
import requests
import os
import mimetypes
from typing import List, Dict, Optional
import asyncio
import aiohttp

# Your existing ExperienceEvaluator and routing functions
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
        # Your existing assessment logic
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

class GuardianGateway:
    """Guardian Gateway: 8 WebPorts + Ulam Routing + MCP Bridge"""
    
    def __init__(self):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        
        # Guardian Configuration
        self.webports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]  # 8 dedicated ports
        self.active_connections = {}
        
        # MCP Bridge to other containers
        self.metatron_url = os.getenv("METATRON_URL", "metatron-router.modal.run")
        self.compactifai_url = os.getenv("COMPACTIFAI_URL", "compactifai-processor.modal.run")
        
        # Your existing routing components
        self.evaluator = ExperienceEvaluator()
        self.ngrok_base_url = os.getenv("NGROK_BASE_URL", "https://5f5295c1f1e6.ngrok-free.app")
        self.media_endpoints = {
            "image": f"{self.ngrok_base_url}/image_to_3d",
            "audio": f"{self.ngrok_base_url}/audio_process", 
            "video": f"{self.ngrok_base_url}/video_process",
            "animation": f"{self.ngrok_base_url}/animation_process",
            "chat": f"{self.ngrok_base_url}/v1/chat/completions"
        }
        self.available_models = [
            {"id": "llama-3.1-70b", "system": "ollama", "type": "llm"},
            {"id": "mistral-7b", "system": "ollama", "type": "llm"},
            {"id": "gpt-4o", "system": "openai", "type": "llm"},
            {"id": "gemma-2", "system": "google", "type": "llm"},
            {"id": "phi-3", "system": "microsoft", "type": "llm"}
        ]

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        # Your existing Ulam routes
        @self.app.get("/")
        async def read_root():
            return {"status": "Guardian Gateway Online", "webports": self.webports}
        
        @self.app.get("/fusion")
        async def get_fusion(size: int = 15):
            nodes = self.prime_spiral_generator(size)
            return {
                'nodes': nodes['nodes'],
                'metatron': [{'x': x, 'y': y} for x, y in zip(np.linspace(-2,2,13), np.linspace(-2,2,13))],
                'fib_spiral': [{'x': x, 'y': y} for x, y in zip(*self.fib_spiral_points(200))]
            }
        
        @self.app.post("/route_media")
        async def route_media(request: Request, file: UploadFile = File(None), query_load: float = 1.0):
            return await self._route_media_internal(request, file, query_load)
        
        # New Guardian-specific routes
        @self.app.get("/guardian/status")
        async def guardian_status():
            return {
                "status": "active",
                "webports_active": len(self.webports),
                "connections": len(self.active_connections),
                "mcp_bridge": "connected"
            }
        
        @self.app.post("/guardian/route_to_compactifai")
        async def route_to_compactifai(model_request: dict):
            """Route model processing to CompactifAI container via MCP"""
            return await self._mcp_route_to_compactifai(model_request)
        
        @self.app.get("/guardian/containers")
        async def get_connected_containers():
            """Get status of all connected containers"""
            return {
                "metatron_router": await self._check_container_health(self.metatron_url),
                "compactifai_processor": await self._check_container_health(self.compactifai_url)
            }

    # Your existing utility functions
    def fib_spiral_points(self, n_points=100):
        theta = np.linspace(0, 4*np.pi, n_points)
        r = np.exp(0.30635 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def ulam_primes_with_one(self, size=15):
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

    def prime_spiral_generator(self, size=15, fib_scale=True):
        grid, prime_grid = self.ulam_primes_with_one(size)
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

    def route_with_prime_weights(self, nodes, query_load: float, media_type: str, selected_model: Optional[str] = None):
        total_weight = sum(node['fib_weight'] * (1.0 if node['value'] == 1 else 0.8 if node['is_prime'] else 0.5)
                           for node in nodes['nodes'])
        assignments = []
        destination = self.media_endpoints.get(media_type, self.ngrok_base_url)
        if media_type == "chat" and selected_model:
            destination = f"{self.ngrok_base_url}/v1/chat/completions?model={selected_model}"
        for node in nodes['nodes']:
            weight = node['fib_weight'] * (1.0 if node['value'] == 1 else 0.8 if node['is_prime'] else 0.5)
            load_share = (weight / total_weight) * query_load
            assignments.append({
                'node': (node['x'], node['y']),
                'load': load_share,
                'destination': destination
            })
        return assignments

    def identify_media_type(self, filename: Optional[str] = None, content_type: Optional[str] = None) -> str:
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

    async def _route_media_internal(self, request: Request, file: UploadFile = File(None), query_load: float = 1.0):
        """Your existing route_media logic"""
        content_type = request.headers.get("Content-Type")
        filename = file.filename if file else None
        media_type = self.identify_media_type(filename, content_type)

        if media_type == "unknown":
            raise HTTPException(status_code=400, detail="Unsupported media type")

        nodes = self.prime_spiral_generator()
        selected_model = None

        if media_type == "chat":
            data = await request.json()
            user_query = data.get("query", "")
            if not user_query:
                raise HTTPException(status_code=400, detail="Chat query required")
            
            evaluations = self.evaluator.evaluate_models(user_query, self.available_models)
            top_model = self.evaluator.get_top_models(evaluations, top_k=1)[0]
            selected_model = top_model["model_id"]
            assignments = self.route_with_prime_weights(nodes, query_load, media_type, selected_model)
            
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
            assignments = self.route_with_prime_weights(nodes, query_load, media_type)
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

    # New Guardian MCP Bridge methods
    async def _mcp_route_to_compactifai(self, model_request: dict):
        """Route model processing to CompactifAI via MCP"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://{self.compactifai_url}/process",
                    json=model_request
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "status": "routed",
                            "processor": "compactifai",
                            "result": result
                        }
                    else:
                        raise HTTPException(status_code=500, detail="CompactifAI processing failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MCP Bridge error: {e}")

    async def _check_container_health(self, container_url: str) -> dict:
        """Check health of connected container"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://{container_url}/health", timeout=5) as response:
                    return {
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "response_time": "ok"
                    }
        except:
            return {"status": "unreachable", "response_time": "timeout"}

    async def start_guardian(self):
        """Start all 8 webport listeners"""
        # In production, this would start multiple uvicorn instances on different ports
        print(f"🛡️ Guardian Gateway starting on {len(self.webports)} webports")
        print(f"🔗 MCP Bridge: Metatron={self.metatron_url}, CompactifAI={self.compactifai_url}")
        
        # Start your existing FastAPI app
        import uvicorn
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

# Modal deployment
app = modal.App("guardian-gateway")

@app.function(
    image=modal.Image.debian_slim().pip_install([
        "fastapi", "uvicorn", "numpy", "networkx", "requests", "aiohttp"
    ]),
    ports=[8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007],  # All 8 webports
    cpu=4.0,
    memory=4096
)
async def deploy_guardian():
    """Deploy the Guardian Gateway"""
    guardian = GuardianGateway()
    await guardian.start_guardian()
    return {"status": "guardian_deployed"}

if __name__ == "__main__":
    # For local development
    guardian = GuardianGateway()
    import uvicorn
    uvicorn.run(guardian.app, host="0.0.0.0", port=8000)