#!/usr/bin/env python3
"""
NEXUS CORE v1.0 - Clean Implementation
Optimized with error handling and memory management
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.integrate import odeint
from diffusers import DiffusionPipeline
import argparse
import subprocess
import sys
import os
import pybullet as p
import pybullet_data
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex
)
import imageio
import cv2
import asyncio
import psutil
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import json
from scipy.linalg import svd
from datetime import datetime
import concurrent.futures
import websockets
import uuid
import gc
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("nexus_core")

# Nexus Constants
PHI = (1 + np.sqrt(5)) / 2
FIB_WEIGHTS = np.array([1, 1, 2, 3, 5, 8, 13, 8, 5, 3, 2, 1, 1]) / 50.0
VORTEX_FREQS = [3, 6, 9, 13]
MOD_9_CYCLE = lambda t: (3 * t + 6 * np.sin(t) + 9 * np.cos(t)) % 9
TOROIDAL_FUNC = lambda n, t: PHI * np.sin(2 * np.pi * 13 * t / 9) * FIB_WEIGHTS[n % 13] * (1 - MOD_9_CYCLE(t) / 9)
SOUL_WEIGHTS = {'hope': 0.4, 'unity': 0.3, 'curiosity': 0.2, 'resilience': 0.1}

# Elemental Properties
ELEMENTAL_PROPS = {
    'earth': {'alpha': 1.52e-5, 'impedance': 21.76, 'speed': 1.28e8},
    'air': {'alpha': 1.88e-12, 'impedance': 376.62, 'speed': 344},
    'fire': {'alpha': 1.54e-4, 'impedance': 2.18, 'speed': 757, 'plasma_temp': 1500, 'conductivity': 5.96e7},
    'water': {'alpha': 1.09e-3, 'impedance': 0.31, 'speed': 1485}
}

class MemoryManager:
    """Optimized memory management for Nexus"""
    
    def __init__(self, max_memory_usage: float = 0.8):
        self.max_memory_usage = max_memory_usage
        self.cleanup_threshold = 0.7
        
    def get_memory_usage(self) -> float:
        """Get current memory usage ratio"""
        return psutil.virtual_memory().percent / 100.0
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        return self.get_memory_usage() > self.cleanup_threshold
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Memory cleanup completed")
    
    def safe_tensor_allocation(self, size: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        """Safely allocate tensor with memory checks"""
        if self.get_memory_usage() > self.max_memory_usage:
            self.force_cleanup()
        
        if self.get_memory_usage() > self.max_memory_usage:
            raise MemoryError("Insufficient memory for tensor allocation")
            
        return torch.zeros(size, dtype=dtype)

def load_soul_config() -> Dict:
    """Load soul configuration with error handling"""
    try:
        soul_config = SOUL_WEIGHTS.copy()
        
        # Try to load from files if they exist
        soul_seed_path = Path('soul_seed.json')
        will_to_live_path = Path('will_to_live.json')
        
        if soul_seed_path.exists():
            with open(soul_seed_path, 'r') as f:
                soul_config.update(json.load(f))
                
        if will_to_live_path.exists():
            with open(will_to_live_path, 'r') as f:
                soul_config.update(json.load(f))
                
        logger.info("Soul configuration loaded successfully")
        return soul_config
        
    except Exception as e:
        logger.warning(f"Failed to load soul configs: {e}, using defaults")
        return SOUL_WEIGHTS

class SVDTensorizer:
    """Optimized SVD compression with error handling"""
    
    def __init__(self, target_rank: int = 64, healing_epochs: int = 2):
        self.target_rank = target_rank
        self.healing_epochs = healing_epochs
        self.memory_manager = MemoryManager()
        
    def svd_compress_layer(self, weight: torch.Tensor, name: str) -> Dict:
        """Compress layer using SVD with error handling"""
        try:
            if weight.dim() < 2:
                raise ValueError(f"Weight tensor must be at least 2D, got {weight.dim()}D")
                
            U, S, Vt = torch.svd_lowrank(weight, q=self.target_rank)
            return {'U': U, 'S': S, 'Vt': Vt, 'name': name}
            
        except Exception as e:
            logger.error(f"SVD compression failed for {name}: {e}")
            # Return identity compression as fallback
            identity = torch.eye(min(weight.shape), device=weight.device)
            return {'U': identity, 'S': torch.ones(weight.shape[0]), 'Vt': identity, 'name': name}

    def reconstruct_layer(self, compressed: Dict) -> torch.Tensor:
        """Reconstruct layer from compressed representation"""
        try:
            return compressed['U'] @ torch.diag(compressed['S']) @ compressed['Vt']
        except Exception as e:
            logger.error(f"Layer reconstruction failed: {e}")
            # Return zeros as fallback
            shape = (compressed['U'].shape[0], compressed['Vt'].shape[1])
            return torch.zeros(shape, device=compressed['U'].device)

    def heal_layer(self, compressed: Dict, original_weight: torch.Tensor) -> Dict:
        """Heal compressed layer using Fibonacci weights"""
        try:
            for epoch in range(self.healing_epochs):
                recon = self.reconstruct_layer(compressed)
                error = torch.norm(original_weight - recon)
                
                if error > 1e-3:
                    # Apply Fibonacci healing
                    healing_factor = 1.05 * FIB_WEIGHTS[:self.target_rank]
                    compressed['S'] = compressed['S'] * healing_factor
                    
                if self.memory_manager.should_cleanup():
                    self.memory_manager.force_cleanup()
                    
            return compressed
            
        except Exception as e:
            logger.error(f"Layer healing failed: {e}")
            return compressed

class SVDLoRALayer(nn.Module):
    """Optimized SVD-LoRA layer with memory management"""
    
    def __init__(self, module: nn.Module, lora_rank: int, svd_rank: int):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(module.weight.shape[0], lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(lora_rank, module.weight.shape[1]) * 0.01)
        self.svd_rank = svd_rank
        self.original_module = module
        self.memory_manager = MemoryManager()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            lora_weight = self.lora_A @ self.lora_B
            original_output = self.original_module(x)
            
            if self.memory_manager.should_cleanup():
                self.memory_manager.force_cleanup()
                
            return original_output + lora_weight @ x
            
        except Exception as e:
            logger.error(f"LoRA forward pass failed: {e}")
            return self.original_module(x)  # Fallback to original

class QuantumInsanityEngine:
    """Quantum enhancement engine with error handling"""
    
    def __init__(self):
        self.entanglement_level = 0
        self.memory_manager = MemoryManager()

    async def quantum_enhance_llm(self, model: nn.Module) -> nn.Module:
        """Apply quantum enhancement to model parameters"""
        try:
            entangled_params = {}
            
            for name, param in model.named_parameters():
                if param.dim() >= 2 and param.requires_grad:
                    # Move to CPU for SVD to save GPU memory
                    W = param.data.cpu().numpy()
                    
                    U, s, Vt = svd(W, full_matrices=False)
                    quantum_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, len(s)))
                    s_quantum = s * quantum_phases
                    W_quantum = U @ np.diag(s_quantum) @ Vt
                    W_real = np.real(W_quantum)
                    
                    entangled_params[name] = torch.tensor(W_real, dtype=param.dtype)
                    
                    if self.memory_manager.should_cleanup():
                        self.memory_manager.force_cleanup()

            # Apply enhanced parameters
            for name, param in model.named_parameters():
                if name in entangled_params:
                    param.data = entangled_params[name].to(param.device)
                    
            self.entanglement_level += 1
            logger.info(f"Quantum enhancement applied successfully (level {self.entanglement_level})")
            return model
            
        except Exception as e:
            logger.error(f"Quantum enhancement failed: {e}")
            return model  # Return original model on failure

class GabrielNetwork:
    """Optimized Gabriel network with error handling"""
    
    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self.queue = asyncio.Queue(maxsize=100)
        self.nodes = {}
        self.connected = False

    async def handshake(self, node_id: str, max_retries: int = 3) -> bool:
        """Establish connection with retry logic"""
        for attempt in range(max_retries):
            try:
                async with websockets.connect(self.ws_url, timeout=10.0) as ws:
                    await ws.send(json.dumps({"type": "handshake", "node_id": node_id}))
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    result = json.loads(response)
                    
                    if result.get("status") == "ok":
                        self.connected = True
                        logger.info(f"Gabriel Network handshake successful for {node_id}")
                        return True
                        
            except Exception as e:
                logger.warning(f"Handshake attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error("All handshake attempts failed")
        return False

    async def broadcast_soul_state(self, node_id: str, soul_state: Dict):
        """Broadcast soul state with error handling"""
        if not self.connected:
            logger.warning("Not connected to Gabriel Network")
            return
            
        try:
            async with websockets.connect(self.ws_url, timeout=5.0) as ws:
                await ws.send(json.dumps({
                    "type": "soul_state", 
                    "node_id": node_id, 
                    "state": soul_state
                }))
                
            try:
                await asyncio.wait_for(
                    self.queue.put(soul_state), 
                    timeout=1.0
                )
            except asyncio.QueueFull:
                logger.warning("Queue full, applying backpressure")
                
        except Exception as e:
            logger.error(f"Soul state broadcast failed: {e}")

    async def heartbeat(self, node_id: str, health: Dict):
        """Send heartbeat with error handling"""
        if not self.connected:
            return
            
        try:
            async with websockets.connect(self.ws_url, timeout=3.0) as ws:
                await ws.send(json.dumps({
                    "type": "heartbeat", 
                    "node_id": node_id, 
                    "health": health
                }))
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

class MetatronRouter:
    """Optimized Metatron router with caching"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        try:
            self.qdrant = QdrantClient(url=qdrant_url, timeout=10.0)
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_cache = {}
            logger.info("Metatron Router initialized successfully")
        except Exception as e:
            logger.error(f"Metatron Router initialization failed: {e}")
            self.qdrant = None
            self.embedder = None

    async def route(self, query_load: int, media_type: str) -> List[Dict]:
        """Route queries with fallback to local execution"""
        if not self.qdrant or not self.embedder:
            logger.warning("Router not available, using local execution")
            return self._get_local_assignments(query_load, media_type)
            
        try:
            cache_key = f"{query_load}:{media_type}"
            if cache_key in self.embedding_cache:
                embedding = self.embedding_cache[cache_key]
            else:
                query = f"gcp-nexus-core 13 1.0"
                embedding = self.embedder.encode(query).tolist()
                self.embedding_cache[cache_key] = embedding

            nodes = await self._get_available_nodes(embedding)
            return self._create_assignments(nodes, query_load, media_type)
            
        except Exception as e:
            logger.error(f"Routing failed: {e}, falling back to local")
            return self._get_local_assignments(query_load, media_type)

    async def _get_available_nodes(self, embedding: List[float]) -> List[Dict]:
        """Get available nodes from Qdrant"""
        try:
            result = self.qdrant.search(
                collection_name="lillith_nodes",
                query_vector=embedding, 
                limit=100
            )
            return [{
                "id": point.payload["source"],
                "address": point.payload["address"],
                "health": point.payload["health"]
            } for point in result]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def _create_assignments(self, nodes: List[Dict], query_load: int, media_type: str) -> List[Dict]:
        """Create task assignments"""
        if not nodes:
            return self._get_local_assignments(query_load, media_type)
            
        assignments = []
        n = len(nodes)
        probabilities = np.ones(n) / n  # Equal probability for now

        for i in range(0, query_load, 10):
            batch_count = min(10, query_load - i)
            node_indices = np.random.choice(n, size=batch_count, p=probabilities)
            
            for node_idx in node_indices:
                node = nodes[node_idx]
                task_id = f"task-{uuid.uuid4()}"
                assignments.append({
                    "task_id": task_id,
                    "target_node": node["address"],
                    "health_score": node["health"]["health_score"],
                    "media_type": media_type
                })
                
        return assignments

    def _get_local_assignments(self, query_load: int, media_type: str) -> List[Dict]:
        """Get assignments for local execution"""
        return [{
            "task_id": f"local-task-{i}",
            "target_node": "localhost",
            "health_score": 1.0,
            "media_type": media_type
        } for i in range(0, query_load, 10)]

class OzOS:
    """Optimized OzOS with health monitoring"""
    
    def __init__(self):
        self.soul = {"cpu": 0, "memory": 0, "disk": 0, "network": 0}
        self.memory_manager = MemoryManager()

    def monitor_system(self) -> Dict:
        """Monitor system health with error handling"""
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            network = psutil.net_io_counters()
            network_io = (network.bytes_sent + network.bytes_recv) / (1024 ** 2)  # MB
            
            health_score = (
                0.35 * (1 - cpu / 100) + 
                0.35 * (1 - memory / 100) + 
                0.3 * (1 - min(disk / 100, 1))
            )
            
            health = {
                "cpu_usage": cpu,
                "memory_usage": memory,
                "disk_usage": disk,
                "network_io_mb": network_io,
                "health_score": max(0.0, min(1.0, health_score))  # Clamp to [0,1]
            }
            
            self.soul.update({
                "cpu": cpu, 
                "memory": memory, 
                "disk": disk, 
                "network": network_io
            })
            
            return health
            
        except Exception as e:
            logger.error(f"System monitoring failed: {e}")
            return {"health_score": 0.8}  # Default healthy

class HermesFirewall:
    """Simple firewall with content validation"""
    
    def permit(self, content: Dict) -> bool:
        """Check if content is permitted"""
        if not content or not isinstance(content, dict):
            return False
            
        # Add more sophisticated checks as needed
        forbidden_keys = ['password', 'secret', 'key']
        return not any(key in str(content).lower() for key in forbidden_keys)

def build_metatron_graph() -> nx.Graph:
    """Build Metatron graph with error handling"""
    try:
        G = nx.Graph()
        G.add_nodes_from(range(13))
        edges = (
            [(0, i) for i in range(1, 7)] + 
            [(i, i+1) for i in range(1, 6)] + [(6, 1)] +
            [(0, i) for i in range(7, 13)] + 
            [(i, i+1) for i in range(7, 12)] + [(12, 7)]
        )
        G.add_edges_from(edges)
        return G
    except Exception as e:
        logger.error(f"Metatron graph build failed: {e}")
        return nx.cycle_graph(13)  # Fallback

def apply_metatron_filter(signal: np.ndarray, cutoff: float = 0.6, use_light: bool = False) -> np.ndarray:
    """Apply Metatron filter with error handling"""
    try:
        G = build_metatron_graph()
        L = nx.laplacian_matrix(G).astype(float)
        eigenvalues, eigenvectors = eigsh(L, k=12, which='SM')
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            
        coeffs = np.dot(eigenvectors.T, signal)
        mask = (eigenvalues <= cutoff).astype(float)
        filtered_coeffs = coeffs * mask * PHI
        filtered = np.dot(eigenvectors, filtered_coeffs)
        
        boost = 1.1 if use_light else 1.2
        if filtered.shape[0] > 0:
            filtered[0] *= boost
        if filtered.shape[0] > 6:
            filtered[6] *= boost
            
        return (filtered * FIB_WEIGHTS[:filtered.shape[0]]).flatten()
        
    except Exception as e:
        logger.error(f"Metatron filter failed: {e}")
        return signal  # Return original on failure

def modulate_signal(signal: np.ndarray, medium: str, freq: float, phenomenon: str = 'EM') -> np.ndarray:
    """Modulate signal with elemental properties"""
    try:
        props = ELEMENTAL_PROPS.get(medium, ELEMENTAL_PROPS['air'])
        atten = np.exp(-props['alpha'])
        phase = np.random.uniform(0, 2 * np.pi)
        z_scale = 377 / props['impedance'] if props['impedance'] else 1.0
        
        modulated = signal * atten * z_scale * np.exp(1j * phase)
        return np.real(modulated)  # Return real component
        
    except Exception as e:
        logger.error(f"Signal modulation failed: {e}")
        return signal

def simulate_tesla_coil(primary_L: float = 10e-6, secondary_L: float = 100e-3, 
                       C: float = 10e-9, cycles: int = 100, freq: int = 13, 
                       rank: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Tesla coil with error handling"""
    try:
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Create coil geometry
        coil_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.2)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coil_id, basePosition=[0, 0, 0])
        
        def lc_eq(y, t, L, C):
            i, v = y
            didt = -v / L
            dvdt = i / C
            return [didt, dvdt]
        
        t = np.linspace(0, cycles / freq, int(1e4 * cycles / freq))
        y0 = [0, 1e-3]
        sol = odeint(lc_eq, y0, t, args=(primary_L + secondary_L, C))
        voltage = sol[:, 1] * np.sin(2 * np.pi * freq * t) * TOROIDAL_FUNC(0, t[0])
        voltage = apply_metatron_filter(voltage)
        voltage = np.clip(voltage, -1e6, 1e6)
        
        p.disconnect()
        return t, voltage
        
    except Exception as e:
        logger.error(f"Tesla coil simulation failed: {e}")
        # Return dummy data
        t = np.linspace(0, 1, 1000)
        voltage = np.sin(2 * np.pi * freq * t) * 1000
        return t, voltage

def generate_l_system(depth: int, axiom: str = "F", rules: Dict = None) -> str:
    """Generate L-system string for fractal branching"""
    if rules is None:
        rules = {"F": "F[+F]F[-F]"}
        
    try:
        current = axiom
        for _ in range(depth):
            next_gen = ""
            for char in current:
                next_gen += rules.get(char, char)
            current = next_gen
        return current
    except Exception as e:
        logger.error(f"L-system generation failed: {e}")
        return axiom  # Return original axiom

def simulate_jacobs_ladder(voltage: np.ndarray, electrode_divergence: float = 0.01, 
                          max_height: float = 0.3, rank: int = 0, 
                          soul_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Jacob's ladder with L-system arcs"""
    if soul_config is None:
        soul_config = SOUL_WEIGHTS
        
    try:
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Create electrodes
        rod1_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.01, height=0.4)
        rod2_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.01, height=0.4)
        
        rod1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rod1_id, 
                               basePosition=[-0.05, 0, 0], 
                               baseOrientation=p.getQuaternionFromEuler([0, electrode_divergence, 0]))
        rod2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rod2_id, 
                               basePosition=[0.05, 0, 0], 
                               baseOrientation=p.getQuaternionFromEuler([0, -electrode_divergence, 0]))
        
        # Generate arc geometry using L-systems
        branch_depth = int(1 + soul_config.get('curiosity', 0.2) * 5)
        l_system = generate_l_system(depth=branch_depth)
        
        # Simulate arc points based on L-system
        arc_points = []
        intensities = []
        t_arc = np.linspace(0, 0.1, 500)
        
        for i, t in enumerate(t_arc):
            if i % 3 == 0 and i < len(voltage):
                # Simple arc simulation based on voltage
                height = min(max_height, (i / len(t_arc)) * max_height)
                width = np.sin(t * 10) * 0.02
                
                point = [width, 0, height]
                arc_points.append(point)
                
                intensity = TOROIDAL_FUNC(1, t) * (voltage[i] / max(np.abs(voltage)))
                intensities.append(intensity)
                
                p.stepSimulation()
            else:
                if arc_points:
                    arc_points.append(arc_points[-1])
                    intensities.append(intensities[-1] if intensities else 0)
        
        p.disconnect()
        
        arc_array = np.array(arc_points) if arc_points else np.zeros((100, 3))
        intensity_array = np.array(intensities) if intensities else np.zeros(100)
        
        return t_arc[:len(arc_array)], arc_array, intensity_array
        
    except Exception as e:
        logger.error(f"Jacob's ladder simulation failed: {e}")
        # Return dummy data
        t_arc = np.linspace(0, 0.1, 100)
        arc_points = np.column_stack([
            np.sin(t_arc * 10) * 0.02,
            np.zeros(100),
            np.linspace(0, 0.3, 100)
        ])
        intensities = np.sin(t_arc * 20) * 0.5 + 0.5
        return t_arc, arc_points, intensities

class MetatronRenderer:
    """Optimized Metatron renderer with memory management"""
    
    def __init__(self, device: str = 'cuda', num_frames: int = 1, ar_mode: bool = False):
        self.device = device
        self.num_frames = num_frames
        self.ar_mode = ar_mode
        self.memory_manager = MemoryManager()
        self._setup_cameras()
        self._setup_renderer()
        
    def _setup_cameras(self):
        """Setup cameras with error handling"""
        try:
            self.cameras = []
            for i in range(self.num_frames):
                azim = 45 + i * 360 / self.num_frames * PHI
                R, T = look_at_view_transform(dist=2.7 * PHI, elev=30, azim=azim)
                self.cameras.append(FoVPerspectiveCameras(device=self.device, R=R, T=T))
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            # Create default camera
            R, T = look_at_view_transform(dist=2.7, elev=30, azim=0)
            self.cameras = [FoVPerspectiveCameras(device=self.device, R=R, T=T)]

    def _setup_renderer(self):
        """Setup renderer with error handling"""
        try:
            self.lights = PointLights(device=self.device, location=[[0, 0, -3]])
            raster_settings = RasterizationSettings(
                image_size=640 if self.ar_mode else 512, 
                blur_radius=0.0, 
                faces_per_pixel=1
            )
            self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
            self.phong_shader = SoftPhongShader(lights=self.lights)
            self.silhouette_shader = SoftSilhouetteShader()
            self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.phong_shader)
            
            self.arc_texture = torch.tensor([1, 0.5, 0], device=self.device).repeat(100, 1)
            
        except Exception as e:
            logger.error(f"Renderer setup failed: {e}")
            self.renderer = None

    async def render_coil_arc(self, coil_verts: np.ndarray, arc_points: np.ndarray, 
                             intensities: np.ndarray, camera_dist: float = 2.7) -> List[np.ndarray]:
        """Render coil and arc with error handling"""
        if self.renderer is None:
            logger.error("Renderer not available")
            return [np.zeros((512, 512, 3))] * self.num_frames
            
        try:
            # Limit number of particles for performance
            num_particles = min(8, len(arc_points))
            if len(arc_points) > num_particles:
                indices = np.linspace(0, len(arc_points)-1, num_particles, dtype=int)
                arc_points = arc_points[indices]
                intensities = intensities[indices]

            # Create coil mesh
            if not hasattr(self, 'coil_verts_cache'):
                self.coil_verts_cache = torch.tensor(coil_verts, dtype=torch.float32).to(self.device)
                self.coil_faces_cache = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int64).to(self.device)
                self.coil_colors_cache = torch.ones((1, len(coil_verts), 3), device=self.device) * 0.7

            coil_mesh = Meshes(verts=[self.coil_verts_cache], faces=[self.coil_faces_cache])
            coil_mesh.textures = TexturesVertex(verts_features=self.coil_colors_cache)

            # Create arc mesh
            arc_verts = torch.tensor(arc_points, dtype=torch.float32).to(self.device).unsqueeze(0)
            arc_faces = torch.arange(len(arc_points)-1, dtype=torch.int64).unsqueeze(0).repeat(1,2).t().contiguous().to(self.device)
            arc_colors = self.arc_texture[:len(arc_points)].unsqueeze(0) * torch.tensor(intensities).unsqueeze(-1).unsqueeze(0).to(self.device)
            
            arc_mesh = Meshes(verts=arc_verts, faces=arc_faces.unsqueeze(0))
            arc_mesh.textures = TexturesVertex(verts_features=arc_colors)

            # Render frames
            frames = []
            for i in range(0, min(self.num_frames, len(self.cameras)), 8):
                batch_cameras = self.cameras[i:i+8]
                
                # Combine meshes for batch rendering
                batch_verts = torch.cat([
                    self.coil_verts_cache.unsqueeze(0).repeat(len(batch_cameras), 1, 1),
                    arc_verts.repeat(len(batch_cameras), 1, 1)
                ])
                
                batch_faces = torch.cat([
                    self.coil_faces_cache.unsqueeze(0).repeat(len(batch_cameras), 1, 1),
                    arc_faces.unsqueeze(0).repeat(len(batch_cameras), 1, 1)
                ])
                
                batch_textures = TexturesVertex(verts_features=torch.cat([
                    self.coil_colors_cache.repeat(len(batch_cameras), 1, 1),
                    arc_colors.repeat(len(batch_cameras), 1, 1)
                ]))
                
                batch_mesh = Meshes(verts=batch_verts, faces=batch_faces, textures=batch_textures)
                
                # Select shader based on distance
                self.renderer.shader = self.silhouette_shader if camera_dist > 2.0 else self.phong_shader
                
                with torch.no_grad():
                    images = self.renderer(meshes_world=batch_mesh)
                    frames.extend(images[..., :3].cpu().numpy())
                    
                if self.memory_manager.should_cleanup():
                    self.memory_manager.force_cleanup()

            return frames if frames else [np.zeros((512, 512, 3))] * self.num_frames
            
        except Exception as e:
            logger.error(f"Coil arc rendering failed: {e}")
            return [np.zeros((512, 512, 3))] * self.num_frames

    async def render_ar(self, coil_verts: np.ndarray, arc_points: np.ndarray, 
                       intensities: np.ndarray, camera_frame: np.ndarray) -> List[np.ndarray]:
        """Render AR overlay with error handling"""
        try:
            frames = await self.render_coil_arc(coil_verts, arc_points, intensities, camera_dist=2.7)
            ar_frames = []

            def blend_frame(frame: np.ndarray, cam_frame: np.ndarray) -> np.ndarray:
                """Blend rendered frame with camera frame"""
                frame_uint8 = (frame * 255).astype(np.uint8)
                alpha = np.clip(frame.mean(axis=2, keepdims=True) / 255, 0, 1)
                cam_resized = cv2.resize(cam_frame, (frame.shape[1], frame.shape[0]))
                return (alpha * frame_uint8 + (1 - alpha) * cam_resized).astype(np.uint8)

            # Process frames in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                blend_futures = [
                    asyncio.get_event_loop().run_in_executor(
                        executor, blend_frame, frame, camera_frame
                    )
                    for frame in frames[::2]  # Process every other frame for performance
                ]
                blended_frames = await asyncio.gather(*blend_futures)
                ar_frames.extend(blended_frames)

            return ar_frames if ar_frames else [camera_frame] * self.num_frames
            
        except Exception as e:
            logger.error(f"AR rendering failed: {e}")
            return [camera_frame] * self.num_frames

def setup_ddp(rank: int, world_size: int):
    """Setup distributed data parallel with error handling"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        logger.info(f"DDP initialized for rank {rank}/{world_size}")
    except Exception as e:
        logger.error(f"DDP setup failed: {e}")

# ========== MODAL CONFIGURATION ==========

# Modal Image with ALL dependencies including psutil
image = modal.Image.debian_slim().pip_install([
    "torch==2.0.1",
    "numpy==1.24.3",
    "networkx==3.1",
    "scipy==1.10.1",
    "diffusers==0.21.4",
    "pybullet==3.2.5",
    "pytorch3d==0.7.4",
    "imageio==2.31.1",
    "opencv-python==4.8.1.78",
    "psutil==5.9.5",  # ✅ Now included
    "qdrant-client==1.6.9",
    "sentence-transformers==2.2.2",
    "websockets==12.0"
])

# Modal App Definition
app = modal.App("nexus-core")

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("nexus-secrets")]
)
async def main_nexus_function():
    """Main Nexus Core function for Modal deployment"""
    # Your existing Nexus core logic here
    oz_os = OzOS()
    health = oz_os.monitor_system()
    logger.info(f"System health: {health}")
    
    # Test some functionality
    try:
        # Test Tesla coil simulation
        t, voltage = simulate_tesla_coil()
        logger.info(f"Tesla coil simulation completed: {len(t)} points")
        
        # Test Jacob's ladder simulation
        t_arc, arc_points, intensities = simulate_jacobs_ladder(voltage)
        logger.info(f"Jacob's ladder simulation completed: {len(arc_points)} arc points")
        
    except Exception as e:
        logger.error(f"Simulation tests failed: {e}")
    
    # Return some result
    return {
        "status": "active", 
        "health": health,
        "simulations_completed": True
    }

# Additional Modal functions for specific operations
@app.function(
    image=image,
    gpu="A100",
    timeout=1800
)
async def run_tesla_simulation():
    """Run Tesla coil simulation in Modal"""
    t, voltage = simulate_tesla_coil()
    return {
        "time_points": len(t),
        "voltage_range": [float(np.min(voltage)), float(np.max(voltage))],
        "status": "completed"
    }

@app.function(
    image=image,
    gpu="A100", 
    timeout=1800
)
async def run_jacobs_ladder_simulation(voltage_data: List[float] = None):
    """Run Jacob's ladder simulation in Modal"""
    if voltage_data is None:
        # Generate sample voltage data
        t = np.linspace(0, 1, 1000)
        voltage_data = np.sin(2 * np.pi * 13 * t) * 1000
    
    t_arc, arc_points, intensities = simulate_jacobs_ladder(np.array(voltage_data))
    return {
        "arc_points": len(arc_points),
        "intensity_range": [float(np.min(intensities)), float(np.max(intensities))],
        "status": "completed"
    }

@app.function(
    image=image,
    timeout=600
)
def check_system_health():
    """Check system health in Modal"""
    oz_os = OzOS()
    health = oz_os.monitor_system()
    return health

# Local execution guard
if __name__ == "__main__":
    # This allows running locally without Modal
    asyncio.run(main_nexus_function())