from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import requests
import subprocess
import psutil
import platform
import os
import json
import time
from typing import Dict, Any, List
import threading

# Import YOUR existing components
from meta_router import MetaRouter
from model_registry import ModelRegistry
from experience_evaluator import ExperienceEvaluator
from turbo_training_orchestrator import TurboTrainingOrchestrator
from proactive_troubleshooter import SystemMonitor
from command_enforcer import CommandEnforcer

# Initialize YOUR existing systems
app = FastAPI(title="Viren MS Enterprise AI", version="1.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing components
model_registry = ModelRegistry({
    "lm_studio": {"url": "http://localhost:1234/v1", "type": "openai_compatible"},
    "ollama": {"url": "http://localhost:11434/api", "type": "ollama"}
})
meta_router = MetaRouter()
experience_evaluator = ExperienceEvaluator()
command_enforcer = CommandEnforcer()
proactive_troubleshooter = SystemMonitor()

# Your Viren Agent (using the one you already have)
class VirenAgent:
    def __init__(self):
        self.id = "viren"
        self.role = "SystemPhysician"
        self.trust_phase = "consultant"
        
    def diagnose_and_repair_system(self, user_complaint: str) -> Dict[str, Any]:
        """YOUR Viren agent with actual command execution"""
        print(f"🩺 Viren: 'Right then, let's have a look at this {user_complaint} situation.'")
        
        # Step 1: Scan actual system
        system_issues = self._scan_actual_system()
        
        # Step 2: Use monitor as information source (but Viren decides)
        system_snapshot = system_monitor.get_system_snapshot()
        ai_analysis = viren_agent.diagnose_with_context(user_complaint, system_snapshot)
        
        # Step 3: Get models and route intelligently
        available_models = model_registry.discover_models()
        if available_models:
            # Use your meta router for intelligent model selection
            selected_model = meta_router.route_query(user_complaint, query_complexity=1.0)
            if selected_model:
                # Use command enforcer to ensure actual execution
                enforced_response = command_enforcer.enforce_troubleshooting(
                    user_complaint, 
                    f"AI Analysis: {ai_analysis}"
                )
            else:
                enforced_response = "Using direct system troubleshooting"
        else:
            enforced_response = "No models available - using built-in troubleshooting"
        
        return {
            "status": "viren_analysis_complete",
            "user_complaint": user_complaint,
            "system_issues_found": system_issues,
            "ai_analysis": ai_analysis,
            "viren_response": enforced_response,
            "models_consulted": len(available_models),
            "british_efficiency": "thoroughly demonstrated",
            "timestamp": time.time()
        }
    
    def _scan_actual_system(self) -> List[Dict[str, Any]]:
        """ACTUAL system scanning"""
        issues = []
        
        try:
            # Real system diagnostics
            if psutil.cpu_percent() > 80:
                issues.append({
                    "type": "high_cpu", 
                    "description": f"CPU usage: {psutil.cpu_percent()}%",
                    "severity": "high"
                })
            
            if psutil.virtual_memory().percent > 85:
                issues.append({
                    "type": "high_memory",
                    "description": f"Memory usage: {psutil.virtual_memory().percent}%",
                    "severity": "high"
                })
            
            if psutil.disk_usage('/').percent > 90:
                issues.append({
                    "type": "low_disk",
                    "description": f"Disk usage: {psutil.disk_usage('/').percent}%",
                    "severity": "medium"
                })
                
        except Exception as e:
            issues.append({
                "type": "scan_error",
                "description": f"Diagnostic error: {str(e)}",
                "severity": "medium"
            })
        
        return issues

# Initialize Viren
viren_agent = VirenAgent()

# Turbo Training in background
turbo_trainer = TurboTrainingOrchestrator()

def start_background_training():
    """Start your turbo training in background"""
    print("🚀 Starting Viren MS evolution in background...")
    time.sleep(5)  # Let the API start first
    turbo_trainer.turbo_train()

# Start background thread
training_thread = threading.Thread(target=start_background_training, daemon=True)
training_thread.start()

# API Routes - Connected to Your React Frontend
@app.get("/")
async def root():
    return {"message": "Viren MS Enterprise AI System - Operational", "status": "ready"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "system": "Viren MS Enterprise AI",
        "timestamp": time.time(),
        "components": {
            "viren_agent": "operational",
            "model_registry": "ready", 
            "meta_router": "active",
            "turbo_training": "running" if turbo_trainer.is_training else "idle"
        }
    }

@app.post("/api/chat")
async def chat_endpoint(request: Dict[str, Any]):
    """Main chat endpoint for your React frontend"""
    try:
        user_message = request.get("message", "")
        
        # Use Viren agent for intelligent response
        viren_result = viren_agent.diagnose_and_repair_system(user_message)
        
        return JSONResponse({
            "response": viren_result["viren_response"],
            "analysis": viren_result["ai_analysis"],
            "issues_found": viren_result["system_issues_found"],
            "status": "success",
            "responder": "Viren MS",
            "timestamp": time.time()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Viren analysis failed: {str(e)}")

@app.get("/api/models")
async def get_models():
    """Get available models for your React frontend"""
    try:
        models = model_registry.discover_models()
        return {
            "available_models": models,
            "count": len(models),
            "systems": ["lm_studio", "ollama"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model discovery failed: {str(e)}")

@app.post("/api/troubleshoot")
async def troubleshoot_system(request: Dict[str, Any]):
    """Advanced troubleshooting endpoint - NOW WITH VIREN IN CHARGE"""
    try:
        user_complaint = request.get("complaint", "")
        
        # VIREN handles it, not some bypassing proactive thing
        result = viren_agent.diagnose_and_repair_system(user_complaint)
        
        return {
            "troubleshooting_result": result,
            "automated_actions_taken": True,
            "viren_in_charge": True,  # This is key
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Troubleshooting failed: {str(e)}")

@app.get("/api/system/status")
async def system_status():
    """Real system status for your React dashboard"""
    try:
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "platform": platform.system(),
            "docker_installed": self._check_docker_installed(),
            "training_stage": turbo_trainer.current_stage,
            "training_cycles": turbo_trainer.training_cycles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/api/training/status")
async def training_status():
    """Get turbo training status for your React frontend"""
    try:
        status = turbo_trainer.get_status()
        return {
            "training_status": status,
            "is_training": turbo_trainer.is_training,
            "current_stage": turbo_trainer.current_stage,
            "progress": f"{min(100, (status['training_cycles'] / status['graduation_requirements']['training_cycles']) * 100):.1f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training status failed: {str(e)}")

def _check_docker_installed(self):
    """Check if Docker is installed"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 VIREN MS ENTERPRISE AI STARTING...")
    print("🔧 Initializing all subsystems...")
    print("🌐 FastAPI running on: http://localhost:8000")
    print("📡 React frontend should connect to: http://localhost:8000/api/chat")
    print("🧠 Turbo training starting in background...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info"
    )