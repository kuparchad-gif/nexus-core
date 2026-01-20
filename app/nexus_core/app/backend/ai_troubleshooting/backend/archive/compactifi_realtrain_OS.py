# local_os_with_training.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.security import HTTPBearer
import json
import asyncio
import os
import uuid
import time
import threading

print("🏠 LOCAL COGNIKUBE OS WITH REAL TRAINING")
print("🦍 Running on THIS machine - no cloud needed!")

# Import the local training function
from local_real_training import train_viren_locally

app = FastAPI(title="Local CogniKube OS")
security = HTTPBearer()

class LocalTrainingManager:
    def __init__(self):
        self.training_status = "ready"
        self.training_results = None
        self.current_training_thread = None
    
    def start_training_async(self):
        """Start training in a separate thread so API doesn't block"""
        if self.training_status == "training":
            return {"status": "error", "message": "Training already in progress"}
        
        self.training_status = "training"
        self.training_results = None
        
        def training_wrapper():
            try:
                result = train_viren_locally()
                self.training_results = {"status": "success", "result": result}
            except Exception as e:
                self.training_results = {"status": "error", "message": str(e)}
            finally:
                self.training_status = "complete"
        
        self.current_training_thread = threading.Thread(target=training_wrapper)
        self.current_training_thread.start()
        
        return {"status": "started", "message": "Viren training started"}

training_manager = LocalTrainingManager()

# Simple authentication (replace with your actual auth)
def verify_token(token: str):
    return {"sub": "admin", "role": "admin"}  # Simple mock

@app.get("/")
async def root():
    return {
        "message": "Local CogniKube OS - Real Training",
        "status": "running",
        "endpoints": {
            "start_training": "POST /train/viren",
            "training_status": "GET /train/status",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "training": training_manager.training_status,
        "location": "local_machine",
        "timestamp": time.time()
    }

@app.post("/train/viren")
async def start_viren_training():
    """Start real Viren training on this machine"""
    print("🎯 MANUAL VIREN TRAINING TRIGGERED LOCALLY")
    
    result = training_manager.start_training_async()
    
    return {
        "training_initiated": True,
        "local_training": True,
        "message": "Viren real training started on local machine",
        "status_endpoint": "/train/status"
    }

@app.get("/train/status")
async def training_status():
    """Check training status"""
    return {
        "training_status": training_manager.training_status,
        "results": training_manager.training_results,
        "timestamp": time.time()
    }

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket for real-time training updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send current training status
            status = {
                "type": "training_status",
                "status": training_manager.training_status,
                "results": training_manager.training_results,
                "timestamp": time.time()
            }
            await websocket.send_json(status)
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        print("Client disconnected")

class LocalCompactifiRuntime:
    """Real training runtime for local execution"""
    
    def optimize_hardware(self):
        print("🦍 LOCAL COMPACTIFAI RUNTIME - REAL TRAINING")
        return train_viren_locally()

# Test endpoint to verify everything works
@app.get("/test/setup")
async def test_setup():
    """Test if training environment is ready"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        return {
            "pytorch_available": True,
            "transformers_available": True,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "training_ready": True
        }
    except ImportError as e:
        return {
            "pytorch_available": False,
            "transformers_available": False,
            "error": str(e),
            "training_ready": False
        }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 STARTING LOCAL COGNIKUBE OS...")
    print("🔗 Local API: http://localhost:8000")
    print("🔗 Training: http://localhost:8000/train/viren")
    print("🔗 Status: http://localhost:8000/train/status")
    print("🔗 WebSocket: ws://localhost:8000/ws/training")
    print("🦍 REAL TRAINING READY - NO CLOUD NEEDED!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)