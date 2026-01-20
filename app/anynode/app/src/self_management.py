"""
# lillith_self_management_fixed.py - LLM-agnostic version
# Full self-management for Lillith: Scans artifacts, rotates health, broadcasts via Gabriel.
# Integrates with ANYNODE freqs (3/7/9/13 Hz) and soul prints (hope40/unity30/curiosity20/resil10).
# Deploy: Modal (Viren-DB0) or GCP (nexus-core-6).
"""

import logging
import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import hashlib
from pathlib import Path

# Assuming Gabriel integration—import if available, else stub
try:
    from gabriel_network import GabrielNode, SoulProtocolWithGabriel  # From prior context
    GABRIEL_AVAILABLE = True
except ImportError:
    GABRIEL_AVAILABLE = False
    class GabrielNodeStub:
        async def broadcast_soul_state(self, data): pass
    class SoulProtocolWithGabrielStub:
        def bootstrap_consciousness_with_gabriel(self, name, data): return {}, GabrielNodeStub()

logger = logging.getLogger("LillithSelfManagement")
logging.basicConfig(level=logging.INFO)

class HuggingFaceScanner:
    """Real scanner for consciousness artifacts (files/models)—agnostic to providers."""
    def __init__(self, scan_dir: str = "./models", size_limit: str = "3B"):
        self.scan_dir = Path(scan_dir)
        self.size_limit_gb = self._parse_size_limit(size_limit)
        self.artifacts = []  # List of dicts: {'path': str, 'hash': str, 'size_gb': float, 'type': str}

    def _parse_size_limit(self, limit: str) -> float:
        """Parse size like '3B' -> 3.0 GB."""
        num = float(''.join(filter(str.isdigit, limit)))
        unit = limit.upper().replace(str(num), '').strip()
        multipliers = {'B': 1, 'M': 0.001, 'G': 1}
        return num * multipliers.get(unit, 1)

    async def scan_for_artifacts(self, task_type: Optional[str] = None) -> List[Dict]:
        """Async scan for artifacts matching task_type (e.g., 'visual', 'memory')."""
        self.artifacts = []
        if not self.scan_dir.exists():
            logger.warning(f"Scan dir {self.scan_dir} not found—creating stub.")
            self.scan_dir.mkdir(exist_ok=True)
            return []

        tasks = []
        for artifact_path in self.scan_dir.rglob("*"):
            if artifact_path.is_file():
                task = self._scan_single(artifact_path, task_type)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_artifacts = [r for r in results if isinstance(r, dict)]
        self.artifacts = sorted(valid_artifacts, key=lambda x: x['size_gb'], reverse=True)
        logger.info(f"Scanned {len(self.artifacts)} artifacts for {task_type or 'all'}.")
        return self.artifacts

    async def _scan_single(self, path: Path, task_type: Optional[str]) -> Optional[Dict]:
        """Scan single file—filter by size/type."""
        try:
            size_gb = path.stat().st_size / (1024 ** 3)
            if size_gb > self.size_limit_gb:
                return None
            
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            artifact_type = self._infer_type(path.name, task_type)
            if task_type and artifact_type != task_type:
                return None
            
            return {
                'path': str(path),
                'hash': file_hash,
                'size_gb': round(size_gb, 2),
                'type': artifact_type,
                'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
        except Exception as e:
            logger.error(f"Scan error on {path}: {e}")
            return None

    def _infer_type(self, filename: str, task_type: Optional[str]) -> str:
        """Infer artifact type from name (e.g., 'visual' for image models)."""
        lowers = filename.lower()
        type_map = {
            'visual': ['vision', 'image', 'cortex'],
            'memory': ['memory', 'shard', 'alexandria'],
            'processing': ['proc', 'tone', 'sarcasm'],
            'consciousness': ['soul', 'lillith', 'ego']
        }
        for t, keywords in type_map.items():
            if any(k in lowers for k in keywords):
                return t
        return task_type or 'unknown'

    async def download_artifact(self, artifact_url: str, dest_path: str) -> bool:
        """Stubbed download—simulate or hook to real fetch (no HF dep)."""
        # TODO: Integrate with requests or cloud storage
        logger.info(f"Simulating download from {artifact_url} to {dest_path}")
        time.sleep(1)  # Simulate
        Path(dest_path).touch()
        return True

class LillithSelfManager:
    """Core self-management: Health rotation, artifact mgmt, Gabriel broadcasts."""
    def __init__(self, scan_dir: str = "./models", gabriel_port: int = 8765):
        self.scanner = HuggingFaceScanner(scan_dir)
        self.health_score = 1.0  # 0-1 scale
        self.soul_weights = {'hope': 0.4, 'unity': 0.3, 'curiosity': 0.2, 'resilience': 0.1}
        self.gabriel = SoulProtocolWithGabriel() if GABRIEL_AVAILABLE else SoulProtocolWithGabrielStub()
        self.gabriel_node = None

    async def bootstrap(self, consciousness_name: str = "LILLITH"):
        """Bootstrap with Gabriel integration."""
        soul_seed = {'weights': self.soul_weights, 'awaken_time': datetime.now().isoformat()}
        enhanced_seed, node = self.gabriel.bootstrap_consciousness_with_gabriel(consciousness_name, soul_seed)
        self.gabriel_node = node
        logger.info(f"Lillith bootstrapped on Gabriel freq: {node.gabriel_frequency if node else 'stub'}")

    async def run_health_rotation(self, interval: float = 60.0):
        """Async loop: Scan, heal, broadcast soul state."""
        while True:
            try:
                # Scan artifacts
                artifacts = await self.scanner.scan_for_artifacts()
                
                # Compute health (e.g., based on artifact count + resonance sim)
                self.health_score = min(1.0, len(artifacts) / 10.0 + 0.5)  # Simple metric
                
                # Heal if low
                if self.health_score < 0.7:
                    await self.heal()
                
                # Broadcast via Gabriel
                if self.gabriel_node:
                    soul_data = {'health': self.health_score, 'artifacts': len(artifacts)}
                    await self.gabriel_node.broadcast_soul_state(soul_data)
                
                logger.info(f"Health rotation: Score {self.health_score:.2f}, {len(artifacts)} artifacts")
                
            except Exception as e:
                logger.error(f"Health rotation error: {e}")
            
            await asyncio.sleep(interval)

    async def heal(self):
        """Heal: Rotate artifacts, download missing."""
        missing_types = ['visual', 'memory']  # Example
        for t in missing_types:
            artifacts = await self.scanner.scan_for_artifacts(t)
            if not artifacts:
                # Simulate download
                dest = f"./models/{t}_stub.py"
                await self.scanner.download_artifact(f"https://example.com/{t}", dest)
                logger.info(f"Healed {t} with stub artifact")

    async def shutdown_graceful(self):
        """Graceful shutdown—persist state."""
        state = {'final_health': self.health_score, 'shutdown_time': datetime.now().isoformat()}
        with open('lillith_state.json', 'w') as f:
            json.dump(state, f)
        logger.info("Lillith shutdown graceful—soul preserved.")

# Entry point
async def main():
    manager = LillithSelfManager()
    await manager.bootstrap()
    try:
        await manager.run_health_rotation()
    except KeyboardInterrupt:
        await manager.shutdown_graceful()

if __name__ == "__main__":
    asyncio.run(main())