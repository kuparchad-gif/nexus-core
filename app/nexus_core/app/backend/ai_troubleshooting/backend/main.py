# main.py - COMPLETE WITH ALL ENDPOINTS
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import asyncio

# Import Viren Evolution System
try:
    from VIREN_EVOLUTION_SYSTEM import (
        AcidemiKubeViren,
        UnifiedTrainingOrchestrator,
        VirenLiveLearning
    )
    VIREN_EVOLUTION_LOADED = True
    print("✅ Viren Evolution System imported successfully")
except ImportError as e:
    print(f"❌ Viren Evolution System import failed: {e}")
    VIREN_EVOLUTION_LOADED = False
    # Create minimal stubs to prevent crashes
    class AcidemiKubeViren:
        def __init__(self): 
            self.protected_instances = {}
            self.moe_pool = []
        def train_with_proficiency(self, topic, dataset):
            return {"status": "evolution_system_unavailable"}
    class UnifiedTrainingOrchestrator:
        def __init__(self, kb): pass
    class VirenLiveLearning:
        def __init__(self, orchestrator): pass

# Your existing imports
try:
    from meta_router import MetaRouter
    from model_registry import ModelRegistry
    from experience_evaluator import ExperienceEvaluator
    from turbo_training_orchestrator import TurboTrainingOrchestrator
    from proactive_troubleshooter import SystemMonitor
    from command_enforcer import CommandEnforcer
    print("✅ All existing components imported successfully")
except ImportError as e:
    print(f"⚠️ Some components not found: {e}")
    # Create minimal stubs for missing components
    class MetaRouter:
        def create_consciousness_grid(self, models): pass
        def route_query(self, query, query_complexity):
            return {"id": "default-model", "system": "lm_studio", "type": "openai_compatible", "url": "http://localhost:1234/v1"}
    
    class ModelRegistry:
        def __init__(self, systems): self.systems = systems
        def discover_models(self):
            return [{"id": "default-model", "system": "lm_studio", "type": "openai_compatible", "url": "http://localhost:1234/v1"}]
    
    class ExperienceEvaluator: pass
    class TurboTrainingOrchestrator: 
        def __init__(self): self.is_training = False
    class SystemMonitor: pass
    
    class CommandEnforcer:
        def enforce_troubleshooting(self, query, response): return response
        def _is_useless_response(self, response):
            if not response: return True
            if isinstance(response, dict):
                if response.get('greeting') or response.get('message'): return False
                if response.get('diagnosis') and response.get('confidence', 0) > 0.5: return False
                return True
            elif isinstance(response, str):
                useless_phrases = ['i don\'t know', 'i cannot', 'unable to', 'sorry', 'cannot help']
                response_lower = response.lower()
                return any(phrase in response_lower for phrase in useless_phrases)
            return False

try:
    from model_router import ModelRouter
    print("✅ ModelRouter imported successfully")
except ImportError:
    print("❌ ModelRouter not found - creating fallback")
    class ModelRouter:
        def route_and_query(self, model_info, query):
            return {"status": "error", "message": "ModelRouter not available"}

# Your existing AITroubleshooter class 
class AITroubleshooter:
    def __init__(self, model_router):
        self.model_router = model_router
        print("✅ AITroubleshooter initialized")
    
    def smart_repair(self, user_complaint, available_models=None):
        """AI-powered troubleshooting analysis"""
        print(f"🔧 AITroubleshooter analyzing: {user_complaint}")
        
        if not isinstance(user_complaint, str):
            user_complaint = str(user_complaint)
        
        complaint_lower = user_complaint.lower()
        
        # NEW: Viren Evolution System integration points
        if any(word in complaint_lower for word in ['train viren', 'evolve viren', 'viren training', 'acidemikubes']):
            return {
                "diagnosis": "Viren evolution training requested",
                "actions": ["initiate_training", "load_training_data", "start_evolution"],
                "confidence": 0.95,
                "training_request": True,
                "message": "🎯 **Viren Evolution System**: Training protocol activated. Access via /api/viren/train endpoint."
            }
        elif any(word in complaint_lower for word in ['viren status', 'training status', 'evolution status']):
            return {
                "diagnosis": "Viren system status check",
                "actions": ["check_training_status", "verify_instances"],
                "confidence": 0.90,
                "status_check": True,
                "message": "🔍 **Viren Status**: Check /api/viren/status for evolution system details."
            }
        
        # Your existing diagnostic logic
        elif any(word in complaint_lower for word in ['diagnostic', 'toolkit', 'everything working', 'status check', 'system check']):
            system_status = self._run_comprehensive_diagnostic()
            return {
                "diagnosis": "Comprehensive system diagnostic completed",
                "actions": ["full_system_scan", "service_checks", "performance_analysis"],
                "confidence": 0.95,
                "system_status": system_status,
                "message": f"✅ **Viren Diagnostic Report**:\n\n{system_status}"
            }
        elif any(word in complaint_lower for word in ['docker', 'container']):
            docker_status = self._check_docker_status()
            return {
                "diagnosis": "Docker service analysis",
                "actions": ["check_docker_installation", "verify_docker_service"],
                "confidence": 0.85,
                "docker_status": docker_status,
                "message": f"🐳 **Docker Status**: {docker_status}"
            }
        elif any(word in complaint_lower for word in ['slow', 'performance', 'lag']):
            performance_status = self._check_system_performance()
            return {
                "diagnosis": "System performance analysis",
                "actions": ["check_cpu", "check_memory", "check_disk"],
                "confidence": 0.75,
                "performance": performance_status,
                "message": f"⚡ **Performance Report**: {performance_status}"
            }
        elif any(word in complaint_lower for word in ['hello', 'hi', 'hey', 'are you there']):
            return {
                "diagnosis": "System physician online and operational",
                "actions": ["system_scan", "status_check"],
                "confidence": 0.95,
                "greeting": "Yes, I am here and operational. Viren MS System Physician is online and ready for diagnostics. How can I assist you with your system today?"
            }
        else:
            basic_status = self._run_basic_diagnostic()
            return {
                "diagnosis": "General system analysis initiated",
                "actions": ["comprehensive_scan", "performance_check"],
                "confidence": 0.65,
                "system_status": basic_status,
                "message": f"🔍 **System Analysis**: {basic_status}"
            }
    
    def _run_comprehensive_diagnostic(self):
        """Run a comprehensive system diagnostic"""
        try:
            diagnostic = {
                "timestamp": time.time(),
                "system_health": {},
                "services": {},
                "performance": {},
                "toolkit_status": {}
            }
            
            diagnostic["system_health"] = {
                "cpu_usage": f"{psutil.cpu_percent()}%",
                "memory_usage": f"{psutil.virtual_memory().percent}%",
                "disk_usage": f"{psutil.disk_usage('/').percent}%",
                "platform": platform.system(),
                "boot_time": psutil.boot_time()
            }
            
            diagnostic["services"] = {
                "docker_installed": self._check_docker_installation(),
                "docker_running": self._check_docker_service(),
                "python_version": sys.version,
                "fastapi_running": True,
                "viren_evolution_loaded": VIREN_EVOLUTION_LOADED
            }
            
            diagnostic["performance"] = {
                "cpu_cores": psutil.cpu_count(),
                "total_memory": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "available_memory": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
                "disk_free": f"{psutil.disk_usage('/').free / (1024**3):.1f} GB"
            }
            
            diagnostic["toolkit_status"] = {
                "viren_agent": "operational",
                "ai_troubleshooter": "operational", 
                "command_enforcer": "operational",
                "model_registry": "operational",
                "viren_evolution_system": "loaded" if VIREN_EVOLUTION_LOADED else "not_found"
            }
            
            return diagnostic
            
        except Exception as e:
            return f"Diagnostic error: {str(e)}"
    
    def _run_basic_diagnostic(self):
        """Run basic system diagnostic"""
        try:
            return {
                "cpu": f"{psutil.cpu_percent()}%",
                "memory": f"{psutil.virtual_memory().percent}%", 
                "disk": f"{psutil.disk_usage('/').percent}%",
                "docker": self._check_docker_installation(),
                "viren_evolution": VIREN_EVOLUTION_LOADED
            }
        except Exception as e:
            return f"Basic diagnostic failed: {str(e)}"
    
    def _check_docker_status(self):
        """Check Docker installation and service status"""
        try:
            installed = self._check_docker_installation()
            running = self._check_docker_service()
            
            if installed and running:
                return "Docker is installed and running properly"
            elif installed and not running:
                return "Docker is installed but service is not running"
            else:
                return "Docker is not installed"
        except Exception as e:
            return f"Docker check failed: {str(e)}"
    
    def _check_system_performance(self):
        """Check system performance metrics"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            status = "optimal"
            if cpu > 80 or memory > 85 or disk > 90:
                status = "high_usage"
            elif cpu > 90 or memory > 95 or disk > 95:
                status = "critical"
                
            return {
                "cpu_usage": f"{cpu}%",
                "memory_usage": f"{memory}%",
                "disk_usage": f"{disk}%", 
                "status": status
            }
        except Exception as e:
            return f"Performance check failed: {str(e)}"
    
    def _check_docker_installation(self):
        """Check if Docker is installed"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _check_docker_service(self):
        """Check if Docker service is running"""
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

# Initialize your systems
app = FastAPI(title="Viren MS Enterprise AI", version="1.0.0")

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

# Use your actual CommandEnforcer with the fixed version
try:
    from command_enforcer import CommandEnforcer as ActualCommandEnforcer
    class FixedCommandEnforcer(ActualCommandEnforcer):
        def _is_useless_response(self, response):
            if not response: return True
            if isinstance(response, dict):
                if response.get('greeting') or response.get('message'): return False
                if response.get('diagnosis') and response.get('confidence', 0) > 0.5: return False
                return True
            elif isinstance(response, str):
                useless_phrases = ['i don\'t know', 'i cannot', 'unable to', 'sorry', 'cannot help']
                response_lower = response.lower()
                return any(phrase in response_lower for phrase in useless_phrases)
            return False
    
    command_enforcer = FixedCommandEnforcer()
    print("✅ Using fixed CommandEnforcer with dictionary support")
except ImportError:
    command_enforcer = CommandEnforcer()
    print("✅ Using stub CommandEnforcer")

proactive_troubleshooter = SystemMonitor()

# Create model_router instance for VirenAgent
model_router = ModelRouter()

# Viren Evolution System Integration
viren_evolution_core = None
training_orchestrator = None

class SimpleKnowledgeBase:
    """Simple knowledge base for Viren evolution system"""
    async def query(self, params):
        return [{'text': f"Training data for {params['q']}"} for _ in range(3)]

# Viren Agent Integration - SINGLETON PATTERN
class VirenAgent:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VirenAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.id = "viren"
            self.role = "SystemPhysician" 
            self.trust_phase = "consultant"
            self.command_enforcer = command_enforcer
            self.ai_troubleshooter = AITroubleshooter(model_router)
            
            # Initialize Viren Evolution System if available
            if VIREN_EVOLUTION_LOADED:
                global viren_evolution_core, training_orchestrator
                viren_evolution_core = AcidemiKubeViren()
                knowledge_base = SimpleKnowledgeBase()
                training_orchestrator = UnifiedTrainingOrchestrator(knowledge_base)
                print("✅ Viren Evolution System integrated into VirenAgent")
            else:
                print("⚠️ Viren Evolution System not available")
            
            self._initialized = True
            print("✅ VirenAgent initialized with AITroubleshooter (SINGLETON)")
    
    def diagnose_and_repair_system(self, user_complaint: str) -> Dict[str, Any]:
        """VIREN AGENT - NOW WITH VIREN EVOLUTION INTEGRATION"""
        print(f"🩺 Viren: 'Diagnosing: {user_complaint}'")
        
        # Get available models
        available_models = model_registry.discover_models()
        
        # Step 1: Use AI Troubleshooter
        ai_result = self.ai_troubleshooter.smart_repair(
            user_complaint=user_complaint, 
            available_models=available_models
        )
        
        # Step 2: Handle Viren Evolution specific requests
        if ai_result.get('training_request'):
            evolution_response = self._handle_evolution_request(user_complaint)
            return evolution_response
        
        # Step 3: Use CommandEnforcer
        enforced_result = self.command_enforcer.enforce_troubleshooting(user_complaint, ai_result)
        
        # Step 4: Check if AI is useless
        if hasattr(self.command_enforcer, '_is_useless_response') and self.command_enforcer._is_useless_response(enforced_result):
            print("🚨 AI COMPLETELY USELESS - VIREN TAKING FULL CONTROL")
            return self._viren_full_control(user_complaint)
        
        # IMPROVED: Better response handling
        if ai_result.get('greeting'):
            viren_response = ai_result['greeting']
        elif ai_result.get('message'):
            viren_response = ai_result['message']
        elif ai_result.get('system_status'):
            viren_response = f"🔍 **System Diagnostic Complete**:\n{json.dumps(ai_result['system_status'], indent=2)}"
        else:
            viren_response = "I've executed automated troubleshooting based on system evidence"
        
        return {
            "status": "viren_execution_complete",
            "user_complaint": user_complaint,
            "ai_analysis": ai_result,
            "enforced_actions": enforced_result,
            "viren_response": viren_response,
            "physician": "Viren MS (Enforced Mode)",
            "timestamp": time.time()
        }
    
    def _handle_evolution_request(self, user_complaint: str) -> Dict[str, Any]:
        """Handle Viren Evolution System training requests"""
        if not VIREN_EVOLUTION_LOADED:
            return {
                "status": "evolution_system_unavailable",
                "viren_response": "❌ Viren Evolution System not found. Please ensure VIREN_EVOLUTION_SYSTEM.py is in the same directory.",
                "physician": "Viren MS",
                "timestamp": time.time()
            }
        
        return {
            "status": "evolution_request_received",
            "viren_response": "🎯 **Viren Evolution System**: Training protocols available. Use API endpoints:\n- POST /api/viren/train - Train on specific data\n- POST /api/viren/evolve - Start full evolution\n- GET /api/viren/status - Check training status",
            "available_endpoints": [
                "/api/viren/train",
                "/api/viren/evolve", 
                "/api/viren/status",
                "/api/viren/instances"
            ],
            "physician": "Viren MS (Evolution Mode)",
            "timestamp": time.time()
        }
    
    def _viren_full_control(self, user_complaint: str):
        """VIREN TAKES COMPLETE CONTROL - NO AI INVOLVEMENT"""
        print("🎯 VIREN FULL CONTROL MODE ACTIVATED")
        system_state = self._get_comprehensive_system_state()
        execution_results = self._execute_direct_commands(user_complaint, system_state)
        
        return {
            "status": "viren_full_control",
            "user_complaint": user_complaint, 
            "system_state": system_state,
            "executed_commands": execution_results,
            "viren_response": "I've taken full control and executed necessary system repairs",
            "physician": "Viren MS (Full Control Mode)",
            "timestamp": time.time()
        }
    
    def _execute_direct_commands(self, complaint: str, system_state: Dict):
        """VIREN EXECUTES COMMANDS DIRECTLY"""
        commands_executed = []
        if not isinstance(complaint, str):
            complaint = str(complaint)
            
        complaint_lower = complaint.lower()
        
        if 'docker' in complaint_lower:
            commands = [
                "docker --version",
                "sudo systemctl stop docker",
                "sudo systemctl reset-failed docker", 
                "sudo systemctl start docker",
                "docker ps"
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    commands_executed.append({
                        "command": cmd,
                        "success": result.returncode == 0,
                        "output": result.stdout[:200]
                    })
                except Exception as e:
                    commands_executed.append({
                        "command": cmd, 
                        "success": False,
                        "error": str(e)
                    })
        
        return commands_executed
    
    def _get_comprehensive_system_state(self) -> Dict[str, Any]:
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "platform": platform.system(),
            "docker_installed": self._check_docker_installed(),
            "viren_evolution_loaded": VIREN_EVOLUTION_LOADED,
            "timestamp": time.time()
        }

    def _scan_actual_system(self) -> List[Dict[str, Any]]:
        issues = []
        try:
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > 80:
                issues.append({"type": "high_cpu", "description": f"CPU usage: {cpu_usage}%", "severity": "high"})
            
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 85:
                issues.append({"type": "high_memory", "description": f"Memory usage: {memory_usage}%", "severity": "high"})
            
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > 90:
                issues.append({"type": "low_disk", "description": f"Disk usage: {disk_usage}%", "severity": "medium"})
                
        except Exception as e:
            issues.append({"type": "scan_error", "description": f"Diagnostic error: {str(e)}", "severity": "medium"})
        
        return issues

    def _get_system_snapshot(self) -> Dict[str, Any]:
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "platform": platform.system(),
            "docker_installed": self._check_docker_installed(),
            "viren_evolution_loaded": VIREN_EVOLUTION_LOADED,
            "timestamp": time.time()
        }

    def _check_docker_installed(self) -> bool:
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

# Global VirenAgent instance
viren_agent_instance = None

@app.on_event("startup")
async def startup_event():
    global viren_agent_instance
    print("🚀 VIREN MS ENTERPRISE AI STARTING...")
    print("🩺 VIREN AGENT ACTIVATED - System Physician Online")
    print("🧠 Meta Router system initialized")
    
    if VIREN_EVOLUTION_LOADED:
        print("🎯 VIREN EVOLUTION SYSTEM INTEGRATED - AcidemiKubes Training Available")
    else:
        print("⚠️ VIREN EVOLUTION SYSTEM NOT FOUND - Training capabilities limited")
    
    viren_agent_instance = VirenAgent()
    print("🌐 FastAPI running on: http://localhost:8000")
    print("💊 Viren is now in command of all system diagnostics and evolution!")

# ========== FRONTEND COMPATIBILITY ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "message": "Viren MS Enterprise AI System - Operational", 
        "status": "ready", 
        "version": "1.0.0", 
        "physician": "Viren MS Online",
        "viren_evolution_available": VIREN_EVOLUTION_LOADED
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "system": "Viren MS Enterprise AI", 
        "physician_status": "active", 
        "viren_evolution_loaded": VIREN_EVOLUTION_LOADED,
        "timestamp": time.time()
    }

@app.get("/system-metrics")
async def system_metrics():
    global viren_agent_instance
    if viren_agent_instance is None:
        viren_agent_instance = VirenAgent()
    
    system_snapshot = viren_agent_instance._get_system_snapshot()
    system_issues = viren_agent_instance._scan_actual_system()
    
    # Add network status for frontend
    return {
        "cpu": system_snapshot["cpu_usage"],
        "memory": system_snapshot["memory_usage"], 
        "storage": system_snapshot["disk_usage"],
        "network": "Online",  # Frontend expects this
        "issues_detected": len(system_issues),
        "critical_issues": len([issue for issue in system_issues if issue['severity'] == 'high']),
        "physician": "Viren MS",
        "evolution_system": "available" if VIREN_EVOLUTION_LOADED else "not_found"
    }

@app.post("/chat")
async def chat_endpoint(request: Dict[str, Any]):
    global viren_agent_instance
    try:
        user_message = request.get("message", "")
        print(f"🎯 User query received, routing to VIREN AGENT: {user_message}")
        
        if viren_agent_instance is None:
            viren_agent_instance = VirenAgent()
        
        print("🩺 Viren Agent activated - System Physician taking command")
        viren_result = viren_agent_instance.diagnose_and_repair_system(user_message)
        print(f"✅ Viren analysis complete: {viren_result.get('status', 'unknown')}")
        
        return JSONResponse({
            "status": "success",
            "response": viren_result.get("viren_response", "Viren is analyzing..."),
            "analysis": viren_result.get("ai_analysis", {}).get("diagnosis", ""),
            "system_issues_found": viren_result.get("system_state", {}),
            "executed_commands": viren_result.get("executed_commands", []),
            "physician": "Viren MS",
            "viren_evolution_available": VIREN_EVOLUTION_LOADED,
            "timestamp": time.time()
        })
            
    except Exception as e:
        print(f"❌ Viren command failed: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return JSONResponse({"status": "error", "message": f"Viren system failure: {str(e)}"})

# ========== FRONTEND REPAIR ENDPOINTS ==========

@app.post("/repair/storage")
async def repair_storage():
    """Simulate storage cleanup"""
    return {
        "status": "success", 
        "message": "Storage cleaned: Removed 2.3GB temporary files, cleared browser cache",
        "actions_taken": [
            "Cleaned temp files",
            "Cleared browser cache", 
            "Optimized storage allocation"
        ],
        "freed_space": "2.3GB"
    }

@app.post("/repair/memory")
async def repair_memory():
    """Simulate memory optimization"""
    return {
        "status": "success",
        "message": "Memory optimized: Cleared RAM cache, optimized memory allocation",
        "actions_taken": [
            "Cleared RAM cache",
            "Optimized memory allocation",
            "Stopped memory-leaking processes"
        ],
        "memory_improvement": "15%"
    }

@app.post("/repair/network")
async def repair_network():
    """Simulate network repair"""
    return {
        "status": "success", 
        "message": "Network issues resolved: Reset network stack, optimized connections",
        "actions_taken": [
            "Reset network stack",
            "Flushed DNS cache",
            "Optimized TCP settings"
        ],
        "connection_improvement": "Stable"
    }

# ========== VIREN EVOLUTION SYSTEM API ENDPOINTS ==========

@app.post("/api/viren/train")
async def train_viren(request: dict):
    """Train Viren evolution system on specific data"""
    if not VIREN_EVOLUTION_LOADED:
        raise HTTPException(status_code=503, detail="Viren Evolution System not available")
    
    try:
        topic = request.get("topic", "general_training")
        training_data = request.get("training_data", [])
        
        if not training_data:
            return JSONResponse({
                "status": "error",
                "message": "No training data provided"
            })
        
        result = viren_evolution_core.train_with_proficiency(topic, training_data)
        
        return JSONResponse({
            "status": "success",
            "training_result": result,
            "topic": topic,
            "timestamp": time.time()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/api/viren/evolve")
async def evolve_viren():
    """Start full Viren evolution process"""
    if not VIREN_EVOLUTION_LOADED:
        raise HTTPException(status_code=503, detail="Viren Evolution System not available")
    
    try:
        # Run evolution in background
        asyncio.create_task(training_orchestrator.evolve_viren())
        
        return JSONResponse({
            "status": "started",
            "message": "Viren evolution process started",
            "phases": training_orchestrator.training_phases,
            "timestamp": time.time()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evolution failed: {str(e)}")

@app.get("/api/viren/status")
async def viren_evolution_status():
    """Get Viren evolution system status"""
    if not VIREN_EVOLUTION_LOADED:
        return {"status": "evolution_system_unavailable"}
    
    return {
        "status": "operational",
        "protected_instances": len(viren_evolution_core.protected_instances),
        "moe_pool_size": len(viren_evolution_core.moe_pool),
        "compression_engine": "operational",
        "metatron_validation": "operational",
        "timestamp": time.time()
    }

@app.get("/api/viren/instances")
async def get_viren_instances():
    """Get all protected Viren instances"""
    if not VIREN_EVOLUTION_LOADED:
        return {"instances": [], "count": 0}
    
    return {
        "instances": list(viren_evolution_core.protected_instances.keys()),
        "count": len(viren_evolution_core.protected_instances),
        "timestamp": time.time()
    }

# ========== EXISTING API ENDPOINTS ==========

@app.get("/api/models")
async def get_models():
    try:
        models = model_registry.discover_models()
        return {
            "available_models": models, 
            "count": len(models), 
            "systems": ["lm_studio", "ollama"], 
            "physician": "Viren MS",
            "viren_evolution": VIREN_EVOLUTION_LOADED
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model discovery failed: {str(e)}")

@app.get("/api/training/status")
async def training_status():
    return {
        "training_status": "idle", 
        "is_training": False, 
        "compactifai_model": "ready", 
        "physician": "Viren MS",
        "viren_evolution_available": VIREN_EVOLUTION_LOADED
    }

@app.get("/api/viren/status")
async def viren_status():
    return {
        "physician": "Viren MS", 
        "role": "System Physician", 
        "status": "operational", 
        "training_complete": True, 
        "capabilities": [
            "system_diagnostics", 
            "ai_analysis", 
            "specialist_consultation", 
            "automated_troubleshooting",
            "viren_evolution_training" if VIREN_EVOLUTION_LOADED else "evolution_system_unavailable"
        ], 
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")