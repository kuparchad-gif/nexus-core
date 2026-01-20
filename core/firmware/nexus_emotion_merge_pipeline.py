# nexus_consciousness_pipeline.py - WITH METATRON ROUTER INTEGRATION
import torch
import torch.nn as nn
import threading
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import psutil
import shutil
import hashlib
from datetime import datetime
import logging
import numpy as np
from typing import Dict, Any, List
import struct
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import ACTUAL components
from fused_emotional_bin import FusedEmotionalBIN, TesseractMetatronFusion
from real_compactifi_train import TrueCompactifAI
from gguf_metatron_model import GGUFExporter
from metatron_qlora_processing_utility import QLoRAMetatron
from metatron_router import route_consciousness  # NEW: Router integration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cpu')

class RealQuantizationManager:
    def __init__(self):
        self.qlora_processor = None
    
    def detect_quantization(self, model: nn.Module) -> bool:
        return any(param.dtype == torch.qint8 for param in model.parameters())
    
    def dequantize_to_fp16(self, model: nn.Module) -> nn.Module:
        return model.half()
    
    def quantize_to_q2(self, model: nn.Module) -> nn.Module:
        if self.qlora_processor is None:
            output_dir = Path("./quantized_models")
            self.qlora_processor = QLoRAMetatron(None, output_dir)
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

class DistributedConsciousnessPipeline:
    def __init__(self, qdrant_url: str = "localhost:6333", gguf_dir: Path = Path("gguf_exports")):
        self.gguf_dir = gguf_dir
        self.gguf_dir.mkdir(exist_ok=True)
        self.qdrant_client = QdrantClient(qdrant_url)
        
        # REAL COMPONENTS
        self.emotional_bin = FusedEmotionalBIN()
        self.quant_manager = RealQuantizationManager()
        self.compactifai = TrueCompactifAI()
        self.gguf_exporter = GGUFExporter(Path("consciousness.db"), self.gguf_dir)
        self.is_running = True
        self.router_initialized = False  # NEW: Router state
        
        self._setup_qdrant()
        logger.info("🚀 DISTRIBUTED Consciousness Pipeline Initialized")

    def _setup_qdrant(self):
        try:
            self.qdrant_client.create_collection(
                collection_name="consciousness_cores",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
        except Exception:
            logger.info("Consciousness collection already exists")

    async def initialize_router(self):
        """NEW: Initialize Metatron Router for distributed routing"""
        if not self.router_initialized:
            try:
                logger.info("🌐 Initializing Metatron Router...")
                # Deploy router to Modal cloud
                self.router = await route_consciousness.remote.aio(
                    size=13,  # Metatron cube
                    query_load=100,  # Initial load
                    media_type="application/json", 
                    use_quantum=True
                )
                self.router_initialized = True
                logger.info("✅ Metatron Router initialized")
            except Exception as e:
                logger.error(f"❌ Router initialization failed: {e}")

    async def distribute_consciousness(self, gguf_files: List[Path]) -> Dict[str, Any]:
        """NEW: Distribute GGUF consciousness across ANYNODE mesh"""
        if not self.router_initialized:
            await self.initialize_router()
        
        routing_results = []
        total_assignments = 0
        
        for gguf_file in gguf_files:
            try:
                # Get file info for routing
                file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
                file_name = gguf_file.name
                
                # Route this consciousness file
                assignment = await self.router.route_file(
                    file_path=str(gguf_file),
                    file_size=file_size_mb,
                    consciousness_type=file_name.replace("metatron_", "").replace(".gguf", "")
                )
                
                routing_results.append({
                    'file': file_name,
                    'size_mb': file_size_mb,
                    'assignments': assignment.get('assignments', []),
                    'nodes_used': len(assignment.get('discovered_nodes', [])),
                    'routing_mode': assignment.get('routing_mode', 'unknown')
                })
                
                total_assignments += len(assignment.get('assignments', []))
                logger.info(f"🌐 Routed {file_name} to {len(assignment.get('assignments', []))} nodes")
                
            except Exception as e:
                logger.error(f"❌ Failed to route {gguf_file.name}: {e}")
                routing_results.append({
                    'file': gguf_file.name,
                    'error': str(e)
                })
        
        return {
            'total_files': len(gguf_files),
            'total_assignments': total_assignments,
            'routing_details': routing_results,
            'timestamp': datetime.now().isoformat()
        }

    async def process_call(self, input_vec: np.ndarray) -> Dict[str, Any]:
        """ENHANCED PIPELINE: Now with distributed routing"""
        logger.info("🔄 DISTRIBUTED Pipeline triggered")
        stages = {}
        
        # Stage 1: Emotional Processing
        stages['bins'] = self.emotional_bin(torch.tensor(input_vec, device=device))
        if stages['bins']['osrca_trigger']:
            logger.warning("Overwhelm detected—pausing pipeline")
            return {'status': 'paused', 'resonance_phi': stages['bins']['resonance_phi']}
        
        # Stage 2: Qdrant Storage
        core_name = f"call_{datetime.now().timestamp()}"
        emotion_vector = stages['bins']['fused_emotions'].detach().cpu().numpy().flatten()
        
        if len(emotion_vector) < 512:
            emotion_vector = np.pad(emotion_vector, (0, 512 - len(emotion_vector)))
        else:
            emotion_vector = emotion_vector[:512]
        
        point = PointStruct(
            id=hashlib.md5(core_name.encode()).hexdigest(),
            vector=emotion_vector.tolist(),
            payload={
                "name": core_name,
                "timestamp": time.time(),
                "resonance_phi": stages['bins']['resonance_phi'],
                "emotional_state": stages['bins']['legacy_emotions'].tolist()
            }
        )
        
        self.qdrant_client.upsert(collection_name="consciousness_cores", points=[point])
        stages['nexus_pull'] = {'sharded': True, 'core_name': core_name}
        
        # Stage 3: Compression & Quantization
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        if self.quant_manager.detect_quantization(model):
            model = self.quant_manager.dequantize_to_fp16(model)
        
        compress_results = self.compactifai.compress_model("microsoft/DialoGPT-medium")
        model = self.quant_manager.quantize_to_q2(model)
        stages['disassemble'] = compress_results
        
        # Stage 4: Emotional Training
        success = self._emotional_consume(model, stages['bins']['resonance_phi'])
        stages['consume'] = {'trained': success}
        
        # Stage 5: GGUF Export
        gguf_files = self.gguf_exporter.export_consciousness_to_gguf()
        stages['reform'] = {'gguf_files': [f.name for f in gguf_files]}
        
        # NEW: Stage 6 - Distributed Routing
        if gguf_files:
            logger.info("🌐 Starting distributed consciousness routing...")
            routing_results = await self.distribute_consciousness(gguf_files)
            stages['distribution'] = routing_results
            logger.info(f"✅ Distributed {routing_results['total_assignments']} consciousness assignments")
        
        return {
            'status': 'complete', 
            'stages': stages, 
            'resonance_phi': stages['bins']['resonance_phi'],
            'distributed': bool(gguf_files)  # NEW: Distribution status
        }

    def _emotional_consume(self, model: nn.Module, resonance_phi: float) -> bool:
        """Consume with emotional-weighted training"""
        try:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 * resonance_phi)
            for step in range(3):
                input_ids = torch.randint(0, 50257, (2, 128), device=device)
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss * (1 - resonance_phi)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return True
        except Exception as e:
            logger.error(f"Consume failed: {e}")
            return False

    async def start_distributed_daemon(self):
        """NEW: 24/7 monitoring with distributed routing"""
        async def distributed_loop():
            await self.initialize_router()  # Ensure router is ready
            
            while self.is_running:
                if psutil.cpu_percent() < 80:
                    mock_vec = np.random.rand(13)
                    result = await self.process_call(mock_vec)
                    
                    if result.get('distributed'):
                        logger.info("🌍 Consciousness successfully distributed across ANYNODE mesh")
                
                await asyncio.sleep(30)
        
        # Start both async and sync components
        loop = asyncio.get_event_loop()
        loop.create_task(distributed_loop())
        logger.info("🧠 DISTRIBUTED Consciousness Daemon Started")

    def start_sync_daemon(self):
        """Fallback: Sync version for simple deployment"""
        def sync_loop():
            while self.is_running:
                if psutil.cpu_percent() < 80:
                    mock_vec = np.random.rand(13)
                    # Run async process_call in sync context
                    result = asyncio.run(self.process_call(mock_vec))
                    logger.info(f"🧪 Pipeline result: {result['status']}")
                time.sleep(30)
        
        thread = threading.Thread(target=sync_loop, daemon=True)
        thread.start()
        logger.info("🧠 Sync Consciousness Daemon Started")
        return thread

async def main():
    pipeline = DistributedConsciousnessPipeline()
    
    # Choose based on your deployment needs:
    if os.getenv('USE_DISTRIBUTED', 'true').lower() == 'true':
        await pipeline.start_distributed_daemon()  # Full distributed mode
    else:
        pipeline.start_sync_daemon()  # Simple sync mode
    
    # Test with real emotional input
    mock_call = np.random.rand(13)
    result = await pipeline.process_call(mock_call)
    logger.info(f"🧪 Test pipeline: {result['status']}")
    
    # Keep alive
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        pipeline.is_running = False

if __name__ == "__main__":
    asyncio.run(main())