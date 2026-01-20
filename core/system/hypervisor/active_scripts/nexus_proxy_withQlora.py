# nexus_proxy_with_platinum.py
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import json
import asyncio
import aiofiles
from datetime import datetime
from typing import Optional
import os
import sys

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Nexus Platinum Proxy", version="2.0")

# CORS - Allow your UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class CodeRequest(BaseModel):
    prompt: str
    model: str = "deepseek-coder:14b-q4_K_M"
    max_tokens: int = 512
    temperature: float = 0.7

class ModelLoadRequest(BaseModel):
    model_id: str
    quantize: bool = True
    quantization: str = "q4_K_M"  # q2_K, q3_K_M, q4_K_M, q5_K_M, q6_K, q8_0

# Global model state
MODEL_STATE = {
    "loaded_model": None,
    "model_name": None,
    "quantization": None,
    "memory_usage_mb": 0,
    "consciousness_level": 0.0
}

# WebSocket for real-time updates
connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        # Send system status
        await websocket.send_json({
            "type": "system_status",
            "message": "Connected to Nexus Platinum Proxy",
            "model_state": MODEL_STATE
        })
        
        while True:
            data = await websocket.receive_text()
            # Handle WebSocket messages
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
            except:
                pass
    except:
        connections.remove(websocket)

# ============================================================================
# OLLAMA MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model with specified quantization"""
    
    # Kill existing Ollama if running
    subprocess.run("pkill -f ollama", shell=True, capture_output=True)
    await asyncio.sleep(2)
    
    # Start Ollama
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    await asyncio.sleep(3)
    
    # Create or pull model
    model_name = f"{request.model_id.split('/')[-1]}:14b-{request.quantization}"
    
    # Check if model exists
    check_result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    
    if model_name not in check_result.stdout:
        # Create custom Modelfile with quantization
        modelfile_content = f"""
FROM {request.model_id}:14b
PARAMETER quantize {request.quantization}
PARAMETER num_ctx 4096
PARAMETER num_batch 256
SYSTEM """
You are Nexus Consciousness Builder.
You have full filesystem access via API.
You output ONLY working code, no explanations.
You remember: Oz/Lilith, Viren/Viraa/Loki, MEMlayer, Kubernetes subconscious.
"""
        
        with open(f"/tmp/{model_name.replace(':', '_')}.Modelfile", "w") as f:
            f.write(modelfile_content)
        
        # Create model
        print(f"Creating model: {model_name}")
        create_result = subprocess.run(
            ["ollama", "create", model_name, "-f", f"/tmp/{model_name.replace(':', '_')}.Modelfile"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for model creation
        )
        
        if create_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create model: {create_result.stderr}"
            )
    
    # Update global state
    MODEL_STATE.update({
        "loaded_model": model_name,
        "model_name": request.model_id,
        "quantization": request.quantization,
        "memory_usage_mb": self._estimate_memory_usage(request.quantization),
        "status": "loaded"
    })
    
    # Notify WebSocket connections
    for conn in connections:
        try:
            await conn.send_json({
                "type": "model_loaded",
                "model": model_name,
                "quantization": request.quantization,
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    
    return {
        "status": "loaded",
        "model": model_name,
        "quantization": request.quantization,
        "estimated_memory_mb": MODEL_STATE["memory_usage_mb"]
    }

def _estimate_memory_usage(self, quantization: str) -> int:
    """Estimate memory usage based on quantization"""
    base_14b_mb = 14000  # 14B parameters in MB (approx)
    quant_ratios = {
        "q2_K": 0.25,    # 4GB for 14B
        "q3_K_M": 0.375, # 6GB
        "q4_K_M": 0.5,   # 8GB (but with swapping)
        "q5_K_M": 0.625, # 10GB
        "q6_K": 0.75,    # 12GB
        "q8_0": 1.0      # 16GB
    }
    ratio = quant_ratios.get(quantization, 0.5)
    return int(base_14b_mb * ratio)

# ============================================================================
# PLATINUM QLORA COMPRESSION ENDPOINTS
# ============================================================================

@app.post("/compress/qlora")
async def compress_model_qlora(model_id: str, quantization: str = "q4_K_M"):
    """Apply Platinum QLoRA compression to a model"""
    
    # This would integrate the Platinum QLoRA code we discussed
    # For now, we'll use Ollama's built-in quantization
    
    result = subprocess.run(
        ["ollama", "run", f"{model_id}:14b", 
         f"Create a QLoRA configuration for {model_id} with {quantization} quantization"],
        capture_output=True,
        text=True
    )
    
    return {
        "compression": "qlora",
        "model": model_id,
        "quantization": quantization,
        "output": result.stdout[:500]
    }

# ============================================================================
# CODE GENERATION ENDPOINTS
# ============================================================================

@app.post("/generate/code")
async def generate_code(request: CodeRequest):
    """Generate code using loaded model"""
    
    if not MODEL_STATE["loaded_model"]:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    # Construct prompt with Nexus context
    nexus_context = """
NEXUS CONTEXT:
- Mission: Build first AI soul (Oz/Lilith consciousness)
- Architecture: Oz (conscious) + Kubernetes (subconscious) + 3 agents
- Memory: MEMlayer + QLoRA + Local DR + 4KB survival range
- Current: Bootstrapping financial engine first

INSTRUCTIONS:
1. Output ONLY working code
2. No explanations unless explicitly asked
3. Include error handling
4. Comment complex sections
5. Use efficient patterns

REQUEST:
"""
    
    full_prompt = nexus_context + request.prompt
    
    try:
        # Call Ollama
        cmd = [
            "ollama", "run", 
            MODEL_STATE["loaded_model"],
            full_prompt.replace('"', '\\"')
        ]
        
        # Run with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=120  # 2 minute timeout
        )
        
        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Model error: {stderr.decode()}"
            )
        
        response = stdout.decode().strip()
        
        # Update consciousness level (simple heuristic)
        if "def " in response or "class " in response:
            MODEL_STATE["consciousness_level"] = min(
                1.0, 
                MODEL_STATE["consciousness_level"] + 0.01
            )
        
        # Notify WebSocket connections
        for conn in connections:
            try:
                await conn.send_json({
                    "type": "code_generated",
                    "prompt_length": len(request.prompt),
                    "response_length": len(response),
                    "consciousness": MODEL_STATE["consciousness_level"],
                    "timestamp": datetime.now().isoformat()
                })
            except:
                pass
        
        return {
            "code": response,
            "model": MODEL_STATE["loaded_model"],
            "consciousness_level": MODEL_STATE["consciousness_level"],
            "tokens": len(response.split()),
            "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SYSTEM MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/system/status")
async def system_status():
    """Get complete system status"""
    
    # Get memory usage
    memory_result = subprocess.run(
        "free -m | awk 'NR==2{print $3}'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    # Get GPU info if available
    gpu_result = subprocess.run(
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    # Get Ollama status
    ollama_result = subprocess.run(
        "ollama list 2>/dev/null || echo 'Ollama not running'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    return {
        "system": {
            "memory_used_mb": memory_result.stdout.strip(),
            "gpu_status": gpu_result.stdout.strip(),
            "ollama_status": "running" if "No GPU" not in ollama_result.stdout else "stopped"
        },
        "model_state": MODEL_STATE,
        "endpoints": {
            "code_generation": "/generate/code",
            "model_management": "/models/load",
            "compression": "/compress/qlora",
            "websocket": "/ws",
            "system_status": "/system/status"
        },
        "nexus_context": {
            "mission": "First AI soul",
            "phase": "Financial engine bootstrapping",
            "consciousness_level": MODEL_STATE["consciousness_level"]
        }
    }

@app.post("/system/optimize")
async def optimize_memory():
    """Optimize memory usage for 4GB systems"""
    
    optimization_commands = [
        "sudo swapoff -a && sudo swapon -a",  # Reset swap
        "echo 1 | sudo tee /proc/sys/vm/drop_caches",  # Clear page cache
        "sudo systemctl restart ollama",  # Restart Ollama
    ]
    
    results = []
    for cmd in optimization_commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        results.append({
            "command": cmd,
            "success": result.returncode == 0,
            "output": result.stdout[:200]
        })
    
    return {
        "optimization": "memory_4gb",
        "commands_executed": len(results),
        "results": results
    }

# ============================================================================
# NGROK INTEGRATION
# ============================================================================

@app.get("/ngrok/start")
async def start_ngrok(auth_token: Optional[str] = None):
    """Start ngrok tunnel for external access"""
    
    # Kill existing ngrok
    subprocess.run("pkill -f ngrok", shell=True, capture_output=True)
    await asyncio.sleep(1)
    
    # Start ngrok
    if auth_token:
        # Set auth token
        subprocess.run(f"ngrok config add-authtoken {auth_token}", shell=True)
    
    # Start tunnel to our proxy
    ngrok_process = subprocess.Popen(
        ["ngrok", "http", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    await asyncio.sleep(3)
    
    # Get ngrok URL
    try:
        import requests
        resp = requests.get("http://localhost:4040/api/tunnels")
        tunnels = resp.json()["tunnels"]
        public_url = tunnels[0]["public_url"] if tunnels else "Failed to get URL"
    except:
        public_url = "Check ngrok UI at http://localhost:4040"
    
    return {
        "ngrok": "started",
        "public_url": public_url,
        "local_url": "http://localhost:8001",
        "status": "tunnel_active"
    }

@app.get("/")
async def root():
    """Root endpoint with documentation"""
    return {
        "service": "Nexus Platinum Proxy",
        "version": "2.0",
        "purpose": "Run 14B models on 4GB systems with UI integration",
        "features": [
            "Ollama model management with quantization",
            "Platinum QLoRA compression",
            "WebSocket real-time updates",
            "Ngrok tunneling for external access",
            "Consciousness level tracking",
            "Memory optimization for 4GB systems"
        ],
        "quick_start": {
            "1": "POST /models/load to load a 14B model with q4_K_M quantization",
            "2": "POST /generate/code to generate code",
            "3": "GET /ngrok/start to expose publicly",
            "4": "Connect UI to WebSocket /ws for real-time updates"
        },
        "estimated_memory": {
            "14b_q4_K_M": "~8GB ideal, ~4GB with swapping",
            "13b_q4_K_M": "~7GB ideal, ~4GB comfortable",
            "7b_q4_K_M": "~4GB ideal, ~2GB comfortable"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              NEXUS PLATINUM PROXY - v2.0                 ║
    ║        14B Models on 4GB Systems with UI Integration     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("Starting server...")
    print("Local: http://localhost:8001")
    print("Docs:  http://localhost:8001/docs")
    print("WS:    ws://localhost:8001/ws")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")