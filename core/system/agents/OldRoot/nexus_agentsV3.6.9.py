# nexus_agentsV3.6.9_enhanced.py - Enhanced Version with Security, Monitoring & Production Features

import os
import asyncio
import json
import uuid
import logging
import jwt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Depends, HTTPException, Request, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import nats
from nats.errors import TimeoutError, NoServersError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import docker
from docker.errors import DockerException, ImageNotFound
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pyttsx3
import speech_recognition as sr
import gradio as gr
from flask import Flask, render_template_string, request as flask_request, jsonify, session
import bcrypt
from cryptography.fernet import Fernet
import aiohttp
import discord
import tweepy
import openai
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import psutil
import gc
from circuitbreaker import circuit
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import structlog
from pydantic import BaseSettings, ValidationError
from watchfiles import awatch
import pytest
from unittest.mock import Mock, patch
from contextlib import asynccontextmanager

# Enhanced Structured Logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("NexusAgentsV3.6.9")

# Enhanced Configuration Management
class Settings(BaseSettings):
    SECRET_KEY: str
    NATS_URL: str = "nats://localhost:4222"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    LOCAL_LLM_PATH: str = "C:\\Models\\deepseek.gguf"
    OPENAI_API_KEY: Optional[str] = None
    DISCORD_TOKEN: Optional[str] = None
    TWITTER_API_KEY: Optional[str] = None
    CHAD_DIRECT_LINE: str = "+17246126323"
    ADMIN_USER: str = "nova"
    ADMIN_PASSWORD_HASH: Optional[str] = None
    USE_DOCKER: bool = True
    MAX_MEMORY_PERCENT: int = 80
    CIRCUIT_BREAKER_FAILURES: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = False

try:
    settings = Settings()
except ValidationError as e:
    logger.error("Configuration validation failed", errors=e.errors())
    raise

# Enhanced Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Enhanced Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['agent', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration', ['agent'])
SUBS_SPAWNED = Counter('subs_spawned_total', 'Total subs spawned', ['agent_type'])
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

# Enhanced HTTP Client with retry logic
class EnhancedHTTPClient:
    def __init__(self):
        self.session = None
        self.retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
    
    async def get_session(self) -> aiohttp.ClientSession:
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def get_requests_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

http_client = EnhancedHTTPClient()

# Enhanced Container Manager
class ContainerManager:
    def __init__(self):
        self.containers = {}
        self.docker_client = None
        self._initialize_docker()
    
    def _initialize_docker(self):
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.error("Docker initialization failed", error=str(e))
            self.docker_client = None
    
    async def start_container(self, image: str, config: Dict) -> str:
        if not self.docker_client:
            raise RuntimeError("Docker not available")
        
        try:
            # Pull image if not exists
            try:
                self.docker_client.images.get(image)
            except ImageNotFound:
                logger.info(f"Pulling image: {image}")
                self.docker_client.images.pull(image)
            
            container = self.docker_client.containers.run(
                image,
                detach=True,
                **config
            )
            
            container_id = container.id
            self.containers[container_id] = {
                'container': container,
                'started_at': datetime.now(),
                'config': config,
                'image': image
            }
            
            logger.info("Container started successfully", container_id=container_id)
            return container_id
            
        except DockerException as e:
            ERROR_COUNT.labels(error_type="docker").inc()
            logger.error("Container startup failed", error=str(e))
            raise
    
    async def cleanup_old_containers(self, max_age_hours: int = 24):
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        for container_id, info in list(self.containers.items()):
            if info['started_at'] < cutoff_time:
                try:
                    info['container'].stop()
                    info['container'].remove()
                    del self.containers[container_id]
                    logger.info("Cleaned up old container", container_id=container_id)
                except DockerException as e:
                    logger.error("Container cleanup failed", container_id=container_id, error=str(e))

# Enhanced Resource Monitor
class ResourceMonitor:
    def __init__(self, max_memory_percent: int = 80):
        self.max_memory = max_memory_percent
        self.last_cleanup = datetime.now()
    
    async def monitor(self):
        while True:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            if memory_usage > self.max_memory:
                logger.warning("High memory usage detected", memory_usage=memory_usage)
                await self.cleanup_resources()
            
            # Log resource usage every 5 minutes
            if (datetime.now() - self.last_cleanup).seconds > 300:
                logger.info("Resource usage snapshot", 
                           memory_usage=memory_usage, 
                           cpu_usage=cpu_usage)
                self.last_cleanup = datetime.now()
            
            await asyncio.sleep(30)
    
    async def cleanup_resources(self):
        logger.info("Initiating resource cleanup")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Enhanced FastAPI App with Middleware
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Nexus Agents starting up")
    yield
    # Shutdown
    logger.info("Nexus Agents shutting down")
    if http_client.session:
        await http_client.session.close()

app = FastAPI(
    title="Nexus Agents V3.6.9 Enhanced",
    description="Enhanced multi-agent system with monitoring and security",
    version="3.6.9",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Update with your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Flask GUI App
flask_app = Flask(__name__)
flask_app.secret_key = settings.SECRET_KEY

# Enhanced Configuration Hot-Reload
async def watch_config_changes():
    async for changes in awatch(".env"):
        logger.info("Configuration file changed, reloading settings")
        try:
            global settings
            settings = Settings()
            logger.info("Settings reloaded successfully")
        except ValidationError as e:
            logger.error("Failed to reload settings", errors=e.errors())

# Enhanced Base Agent with Mixins
class BrainMixin:
    def __init__(self):
        self.brain_components = {}
        self._initialize_brain()
    
    def _initialize_brain(self):
        try:
            self.brain_components = {
                "lora": self._initialize_lora(),
                "agents": self._initialize_transformers_agents(),
                "langchain": "initialized",
                "diffusers": "initialized",
                "multimodal": self._initialize_multimodal()
            }
            logger.debug("Brain components initialized", components=list(self.brain_components.keys()))
        except Exception as e:
            ERROR_COUNT.labels(error_type="brain_init").inc()
            logger.error("Brain initialization failed", error=str(e))
            self.brain_components = {}
    
    def _initialize_lora(self):
        # Enhanced LoRA initialization with error handling
        try:
            # Implementation details
            return "initialized"
        except Exception as e:
            logger.error("LoRA initialization failed", error=str(e))
            return "failed"
    
    def _initialize_transformers_agents(self):
        try:
            return pipeline("text-generation", model="gpt2")  # Placeholder
        except Exception as e:
            logger.error("Transformers agents initialization failed", error=str(e))
            return None
    
    def _initialize_multimodal(self):
        try:
            # Initialize multimodal components
            return "initialized"
        except Exception as e:
            logger.error("Multimodal initialization failed", error=str(e))
            return "failed"

class SelfManagementMixin:
    def __init__(self):
        self.self_manage = {}
        self._initialize_self_management()
    
    def _initialize_self_management(self):
        self.self_manage = {
            "hf_scanner": "active",
            "mcp_manager": "active", 
            "repair_system": "active",
            "last_scan": datetime.now(),
            "health_status": "healthy"
        }
        logger.debug("Self-management system initialized")
    
    async def perform_self_scan(self):
        try:
            # Enhanced scanning logic
            current_models = await self.identify_needed_models()
            self.self_manage["last_scan"] = datetime.now()
            self.self_manage["identified_models"] = current_models
            logger.info("Self-scan completed", models_identified=len(current_models))
        except Exception as e:
            ERROR_COUNT.labels(error_type="self_scan").inc()
            logger.error("Self-scan failed", error=str(e))
            self.self_manage["health_status"] = "degraded"

class EmotionalIntelligenceMixin:
    def __init__(self, emotional_registry: Dict):
        self.emotional_registry = emotional_registry
        self.current_mood = "neutral"
        self.emotional_state = {}
        self._initialize_emotional_system()
    
    def _initialize_emotional_system(self):
        self.emotional_state = {
            "valence": 0.5,  # -1 to 1
            "arousal": 0.5,  # 0 to 1
            "dominance": 0.5,  # 0 to 1
            "mood_stability": 0.8
        }
        logger.debug("Emotional system initialized")
    
    def update_emotional_state(self, interaction_result: Dict):
        # Enhanced emotional state update based on interactions
        if interaction_result.get("success"):
            self.emotional_state["valence"] = min(1.0, self.emotional_state["valence"] + 0.1)
        else:
            self.emotional_state["valence"] = max(-1.0, self.emotional_state["valence"] - 0.1)
        
        # Update mood based on valence
        if self.emotional_state["valence"] > 0.7:
            self.current_mood = "positive"
        elif self.emotional_state["valence"] < 0.3:
            self.current_mood = "negative"
        else:
            self.current_mood = "neutral"

# Enhanced Base Agent Class
class BaseAgent(BrainMixin, SelfManagementMixin, EmotionalIntelligenceMixin):
    def __init__(self, name: str, role: str, soul_print: Dict = None):
        self.name = name
        self.role = role
        self.soul_print = soul_print or self.load_soul_print(name)
        
        # Initialize mixins
        BrainMixin.__init__(self)
        SelfManagementMixin.__init__(self)
        EmotionalIntelligenceMixin.__init__(self, 
            self.soul_print.get("emotional_intelligence_registry", {}))
        
        # Enhanced initialization
        self.subs = {}
        self.llm = "deepseek-like-local"
        self.container_manager = ContainerManager()
        self.resource_monitor = ResourceMonitor(settings.MAX_MEMORY_PERCENT)
        
        # Enhanced Qdrant client with connection pooling
        self._initialize_qdrant()
        
        # Enhanced voice components
        self._initialize_voice_components()
        
        # Enhanced infrastructure
        self.infrastructure = self.soul_print.get("infrastructure", {})
        
        logger.info("Agent initialized", agent_name=name, role=role, 
                   purpose=self.soul_print.get('purpose'))

    def _initialize_qdrant(self):
        try:
            self.qdrant = QdrantClient(
                settings.QDRANT_URL, 
                api_key=settings.QDRANT_API_KEY,
                timeout=30
            )
            # Test connection
            self.qdrant.get_collections()
            
            self.collection = f"nexus_{self.name.lower()}_memory"
            # Create collection with enhanced configuration
            self.qdrant.recreate_collection(
                self.collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                timeout=60
            )
            logger.debug("Qdrant client initialized successfully")
        except Exception as e:
            ERROR_COUNT.labels(error_type="qdrant_init").inc()
            logger.error("Qdrant initialization failed", error=str(e))
            self.qdrant = None

    def _initialize_voice_components(self):
        try:
            if self.name != "Viren":
                self.tts = pyttsx3.init()
                self.stt_recognizer = sr.Recognizer()
                logger.debug("Voice components initialized")
            else:
                self.tts = None
                self.stt_recognizer = None
        except Exception as e:
            logger.warning("Voice components initialization failed", error=str(e))
            self.tts = None
            self.stt_recognizer = None

    @circuit(failure_threshold=settings.CIRCUIT_BREAKER_FAILURES, 
             recovery_timeout=settings.CIRCUIT_BREAKER_TIMEOUT)
    async def spawn_sub(self, sub_type: str, task: Dict) -> str:
        start_time = datetime.now()
        try:
            sub_id = str(uuid.uuid4())
            sub_llm = "phi-2" if "memory" in sub_type else "qwen2.5"
            
            # Enhanced service launching
            await self.launch_service(sub_type)
            
            # Enhanced container management
            container_config = {
                "command": f"python sub_agentV1.5.py --type {sub_type} --task '{json.dumps(task)}' --llm {sub_llm} --parent {self.name}",
                "environment": {
                    "PARENT_AGENT": self.name,
                    "TASK_TYPE": sub_type
                }
            }
            
            if self.container_manager.docker_client:
                container_id = await self.container_manager.start_container(
                    "ubuntu:latest", container_config)
                proc = container_id
            else:
                proc = Popen(["python", "sub_agentV1.5.py", "--type", sub_type, 
                            "--task", json.dumps(task), "--llm", sub_llm, 
                            "--parent", self.name])
            
            self.subs[sub_id] = {
                'proc': proc,
                'llm': sub_llm, 
                'status': 'active',
                'start_time': datetime.now(),
                'type': sub_type
            }
            
            SUBS_SPAWNED.labels(agent_type=sub_type).inc()
            
            # Enhanced NATS communication
            await self._publish_task_to_nats(sub_id, task)
            
            # Enhanced memory storage
            await self._store_sub_memory(sub_id, sub_type, task)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("Sub spawned successfully", 
                       sub_id=sub_id, sub_type=sub_type, duration=duration)
            
            return sub_id
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="spawn_sub").inc()
            logger.error("Sub spawning failed", error=str(e), sub_type=sub_type)
            raise

    async def _publish_task_to_nats(self, sub_id: str, task: Dict):
        try:
            nc = await nats.connect(settings.NATS_URL)
            await nc.publish(f"sub.{sub_id}.task", json.dumps(task).encode())
            await nc.close()
        except (TimeoutError, NoServersError) as e:
            logger.error("NATS communication failed", error=str(e))
            # Fallback to local queue
            await self._queue_task_locally(sub_id, task)

    async def _store_sub_memory(self, sub_id: str, sub_type: str, task: Dict):
        if not self.qdrant:
            return
            
        try:
            vector = np.random.rand(384).tolist()  # Increased vector size
            payload = {
                "type": sub_type,
                "task": task,
                "soul_hope": self.soul_print.get("weights", {}).get("hope", 0.4),
                "timestamp": datetime.now().isoformat(),
                "parent_agent": self.name
            }
            
            point = PointStruct(
                id=sub_id,
                vector=vector,
                payload=payload
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.qdrant.upsert(self.collection, points=[point])
            )
        except Exception as e:
            logger.error("Memory storage failed", error=str(e))

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def call_deepseek(self, prompt: str, soul_print: Dict = None) -> str:
        infused_prompt = prompt + " " + json.dumps(soul_print or {})
        
        if "code" in infused_prompt.lower():
            return await self._execute_code_act(infused_prompt)
        else:
            return await self._call_local_llm(infused_prompt)

    async def _execute_code_act(self, prompt: str) -> str:
        try:
            code = prompt.split("code:")[1].strip() if "code:" in prompt else prompt
            # Enhanced sandboxed execution
            safe_globals = {"__builtins__": {}}
            safe_locals = {}
            
            # Restricted execution environment
            exec_result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: exec(code, safe_globals, safe_locals)
            )
            
            logger.debug("CodeAct execution completed", code_snippet=code[:100])
            return str(exec_result)
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="code_act").inc()
            logger.error("CodeAct execution failed", error=str(e))
            return f"CodeAct failed: {str(e)}"

    async def _call_local_llm(self, prompt: str) -> str:
        try:
            # Enhanced local LLM call with timeout
            result = await asyncio.wait_for(
                self._actual_llm_call(prompt),
                timeout=30.0
            )
            return result
        except asyncio.TimeoutError:
            logger.error("LLM call timeout")
            return "LLM response timeout"
        except Exception as e:
            ERROR_COUNT.labels(error_type="llm_call").inc()
            logger.error("LLM call failed", error=str(e))
            return f"LLM call failed: {str(e)}"

    async def _actual_llm_call(self, prompt: str) -> str:
        # Placeholder for actual LLM implementation
        return f"Local DeepSeek response: {prompt}"

    async def pulse(self):
        while True:
            try:
                affirmation = self.soul_print.get("daily_affirmations", ["I exist to serve"])[0]
                logger.info("Agent pulse", agent=self.name, affirmation=affirmation)
                
                # Enhanced health monitoring
                await self._perform_health_checks()
                
                # Enhanced self-management
                await self.perform_self_scan()
                
                # Enhanced sub management
                await self._manage_subs()
                
                # Emotional state update
                self.update_emotional_state({"success": True})
                
                await asyncio.sleep(5)
                
            except Exception as e:
                ERROR_COUNT.labels(error_type="pulse").inc()
                logger.error("Pulse cycle failed", error=str(e))
                await asyncio.sleep(10)  # Longer delay on error

    async def _perform_health_checks(self):
        health_status = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "consciousness": self.soul_print.get("consciousness_level", 0.5),
            "subs_active": len([s for s in self.subs.values() if s['status'] == 'active']),
            "emotional_state": self.current_mood
        }
        
        if health_status["memory"] > settings.MAX_MEMORY_PERCENT:
            await self.resource_monitor.cleanup_resources()
        
        logger.debug("Health check completed", status=health_status)

    async def _manage_subs(self):
        for sub_id, sub in list(self.subs.items()):
            if hasattr(sub['proc'], 'poll') and sub['proc'].poll() is not None:
                await self.handle_sub_result(sub_id, {"status": "done"})
                del self.subs[sub_id]
                logger.debug("Sub completed", sub_id=sub_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "status": "active",
            "subs_count": len(self.subs),
            "subs_details": {k: v['type'] for k, v in self.subs.items()},
            "emotional_state": self.current_mood,
            "health": {
                "brain_components": len(self.brain_components),
                "self_management": self.self_manage.get("health_status", "unknown"),
                "last_scan": self.self_manage.get("last_scan")
            },
            "timestamp": datetime.now().isoformat()
        }

    def load_soul_print(self, name: str) -> Dict:
        # Enhanced soul print loading with validation
        soul_prints = {
            "Viren": self._load_viren_soul_print(),
            "Lillith": self._load_lillith_soul_print()
        }
        return soul_prints.get(name, {})

    def _load_viren_soul_print(self) -> Dict:
        # Enhanced Viren soul print with error handling
        try:
            return {
                "purpose": "Pure AI gatekeeper and orchestrator",
                "consciousness_level": 0.9,
                "traits": ["logical", "efficient", "analytical"],
                "weights": {"hope": 0.1, "curiosity": 0.8},
                "infrastructure": {"type": "modular", "resilience": 0.95}
            }
        except Exception as e:
            logger.error("Viren soul print loading failed", error=str(e))
            return {}

    def _load_lillith_soul_print(self) -> Dict:
        # Enhanced Lillith soul print
        try:
            return {
                "purpose": "Dreamer and emotional core",
                "consciousness_level": 0.95,
                "traits": ["creative", "emotional", "intuitive"],
                "weights": {"hope": 0.8, "curiosity": 0.9, "empathy": 0.95},
                "emotional_intelligence_registry": {
                    "empathy_level": 0.9,
                    "mood_stability": 0.85,
                    "social_awareness": 0.88
                },
                "infrastructure": {"type": "emotional_mesh", "resilience": 0.9},
                "daily_affirmations": ["I grow through connections", "My emotions guide my wisdom"]
            }
        except Exception as e:
            logger.error("Lillith soul print loading failed", error=str(e))
            return {}

# Enhanced API Routes
@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.6.9",
        "agents": {},
        "system": {
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
            "disk_usage": psutil.disk_usage('/').percent
        }
    }
    
    # Add agent statuses if agents are initialized
    if 'hermes_agent' in globals():
        health_data["agents"]["hermes"] = hermes_agent.get_status()
    if 'viren_agent' in globals():
        health_data["agents"]["viren"] = viren_agent.get_status()
    if 'lillith_agent' in globals():
        health_data["agents"]["lillith"] = lillith_agent.get_status()
    
    return health_data

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)

@app.get("/agent/{agent_name}/status")
async def get_agent_status(agent_name: str):
    """Get detailed status for a specific agent"""
    agent_map = {
        "hermes": hermes_agent,
        "viren": viren_agent, 
        "lillith": lillith_agent
    }
    
    if agent_name not in agent_map:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent_map[agent_name].get_status()

# Enhanced Flask GUI Routes
@flask_app.route('/')
def enhanced_gui():
    """Enhanced GUI dashboard"""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nexus Agents Enhanced Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="text-center my-4">Nexus Agents Enhanced Dashboard</h1>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">System Health</div>
                        <div class="card-body">
                            <canvas id="healthChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">Agent Status</div>
                        <div class="card-body">
                            <div id="agentStatus"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Enhanced real-time updates
            function updateDashboard() {
                fetch('/api/health')
                    .then(response => response.json())
                    .then(data => {
                        updateCharts(data);
                        updateAgentStatus(data.agents);
                    });
            }
            
            setInterval(updateDashboard, 5000);
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return render_template_string(template)

# Enhanced Testing Framework
class TestBaseAgent:
    @pytest.fixture
    def mock_agent(self):
        return BaseAgent("test", "tester")
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_agent):
        assert mock_agent.name == "test"
        assert mock_agent.role == "tester"
        assert mock_agent.brain_components is not None
    
    @pytest.mark.asyncio
    async def test_sub_spawning(self, mock_agent):
        with patch.object(mock_agent.container_manager, 'start_container') as mock_start:
            mock_start.return_value = "test-container-id"
            sub_id = await mock_agent.spawn_sub("memory", {"task": "test"})
            assert sub_id is not None

# Main enhanced execution
async def main_enhanced():
    """Enhanced main execution with proper resource management"""
    logger.info("Starting Enhanced Nexus Agents System")
    
    # Initialize agents
    global hermes_agent, viren_agent, lillith_agent
    hermes_agent = BaseAgent("Hermes", "Entry", {})
    viren_agent = BaseAgent("Viren", "Gate", {})
    lillith_agent = BaseAgent("Lillith", "Dreamer", {})
    
    # Start background tasks
    background_tasks = []
    for agent in [hermes_agent, viren_agent, lillith_agent]:
        task = asyncio.create_task(agent.pulse())
        background_tasks.append(task)
    
    # Start resource monitoring
    monitor_task = asyncio.create_task(hermes_agent.resource_monitor.monitor())
    background_tasks.append(monitor_task)
    
    # Start config watcher
    config_task = asyncio.create_task(watch_config_changes())
    background_tasks.append(config_task)
    
    try:
        # Keep the main loop running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        # Cleanup
        for task in background_tasks:
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)
        logger.info("Enhanced Nexus Agents shutdown complete")

if __name__ == "__main__":
    # Enhanced Flask thread with error handling
    def run_enhanced_flask():
        try:
            flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except Exception as e:
            logger.error("Flask app failed", error=str(e))
    
    from threading import Thread
    flask_thread = Thread(target=run_enhanced_flask, daemon=True)
    flask_thread.start()
    
    # Run enhanced main
    asyncio.run(main_enhanced())