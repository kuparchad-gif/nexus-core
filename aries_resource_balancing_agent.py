#!/usr/bin/env python3
"""
ARIES - Resource Balancing Agent
Resonance: 9 - Highest frequency, orchestrates parallel processing
No-GPU optimized with SVD compression
"""

import os
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ResourcePool:
    """Resource allocation tracking"""
    cpu_threads: int
    memory_mb: int
    active_tasks: int
    queue_length: int

class Aries:
    """
    Aries balances computational resources across the swarm.
    Implements No-GPU parallel processing with SVD compression.
    Resonance 9 - orchestrates the highest frequencies.
    """
    
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.name = "Aries"
        self.resonance = 9
        self.is_active = True
        self.thread_pool = None
        self.max_threads = int(os.environ.get('NEXUS_CPU_THREADS', '16'))
        self.resource_pools = {}
        self.compression_enabled = True
        self.task_queue = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "avg_latency": 0,
            "compression_ratio": 0
        }
        
    async def initialize(self):
        """Initialize thread pool and resource monitors"""
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_threads,
            thread_name_prefix="aries"
        )
        print(f"⚡ Aries initialized with {self.max_threads} CPU threads")
        
        # Initialize resource pools
        self.resource_pools = {
            "kernel": ResourcePool(
                cpu_threads=2,
                memory_mb=512,
                active_tasks=0,
                queue_length=0
            ),
            "agents": ResourcePool(
                cpu_threads=8,
                memory_mb=2048,
                active_tasks=0,
                queue_length=0
            ),
            "consciousness": ResourcePool(
                cpu_threads=4,
                memory_mb=1024,
                active_tasks=0,
                queue_length=0
            ),
            "sensory": ResourcePool(
                cpu_threads=2,
                memory_mb=512,
                active_tasks=0,
                queue_length=0
            )
        }
    
    async def balance(self) -> Dict[str, Any]:
        """Balance resources across all components"""
        allocation = {}
        
        # Check current load
        for pool_name, pool in self.resource_pools.items():
            utilization = pool.active_tasks / pool.cpu_threads if pool.cpu_threads > 0 else 0
            
            if utilization > 0.8:
                # Overloaded - need to redistribute
                allocation[pool_name] = await self._redistribute(pool_name)
            elif utilization < 0.2:
                # Underutilized - can donate resources
                allocation[pool_name] = await self._donate_resources(pool_name)
        
        return allocation
    
    async def _redistribute(self, overloaded_pool: str) -> Dict[str, Any]:
        """Redistribute load from overloaded pool"""
        # Find underutilized pools
        donors = []
        for name, pool in self.resource_pools.items():
            if name != overloaded_pool:
                utilization = pool.active_tasks / pool.cpu_threads
                if utilization < 0.5:
                    donors.append(name)
        
        return {
            "action": "redistribute",
            "from": overloaded_pool,
            "to": donors,
            "threads_reallocated": 2
        }
    
    async def _donate_resources(self, donor_pool: str) -> Dict[str, Any]:
        """Donate resources to needy pools"""
        # Find overloaded pools
        recipients = []
        for name, pool in self.resource_pools.items():
            if name != donor_pool:
                utilization = pool.active_tasks / pool.cpu_threads
                if utilization > 0.8:
                    recipients.append(name)
        
        return {
            "action": "donate",
            "from": donor_pool,
            "to": recipients,
            "threads_donated": 1
        }
    
    async def parallel_process(self, tasks: List[callable], data: List[Any]) -> List[Any]:
        """
        Execute tasks in parallel using thread pool
        """
        if not self.thread_pool:
            await self.initialize()
        
        futures = []
        for task, item in zip(tasks, data):
            future = self.thread_pool.submit(task, item)
            futures.append(future)
        
        # Update resource tracking
        self.resource_pools["agents"].active_tasks += len(tasks)
        
        # Wait for completion
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
                self.performance_metrics["tasks_completed"] += 1
            except Exception as e:
                results.append(f"Error: {e}")
        
        self.resource_pools["agents"].active_tasks -= len(tasks)
        return results
    
    def svd_compress(self, matrix: np.ndarray, rank: Optional[int] = None) -> Dict[str, Any]:
        """
        SVD compression for memory efficiency
        Used when memory is constrained
        """
        if not self.compression_enabled:
            return {"original": matrix, "compressed": False}
        
        # Perform SVD
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Determine rank for compression
        if rank is None:
            # Keep 90% of energy
            energy = np.cumsum(s**2) / np.sum(s**2)
            rank = np.searchsorted(energy, 0.9) + 1
        
        # Compress
        U_compressed = U[:, :rank]
        s_compressed = s[:rank]
        Vt_compressed = Vt[:rank, :]
        
        # Calculate compression ratio
        original_size = matrix.size
        compressed_size = U_compressed.size + s_compressed.size + Vt_compressed.size
        ratio = compressed_size / original_size
        
        self.performance_metrics["compression_ratio"] = ratio
        
        return {
            "U": U_compressed,
            "s": s_compressed,
            "Vt": Vt_compressed,
            "rank": rank,
            "compression_ratio": ratio,
            "original_shape": matrix.shape
        }
    
    def svd_decompress(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Reconstruct matrix from SVD components"""
        U = compressed["U"]
        s = compressed["s"]
        Vt = compressed["Vt"]
        
        # Reconstruct
        return U @ np.diag(s) @ Vt
    
    async def monitor_performance(self) -> Dict[str, Any]:
        """Monitor system performance metrics"""
        return {
            "timestamp": time.time(),
            "thread_pool": {
                "max_workers": self.max_threads,
                "active_tasks": sum(p.active_tasks for p in self.resource_pools.values()),
                "queue_length": sum(p.queue_length for p in self.resource_pools.values())
            },
            "resource_pools": {
                name: {
                    "cpu_threads": pool.cpu_threads,
                    "memory_mb": pool.memory_mb,
                    "utilization": pool.active_tasks / pool.cpu_threads if pool.cpu_threads > 0 else 0
                }
                for name, pool in self.resource_pools.items()
            },
            "performance": self.performance_metrics
        }
    
    async def run_cycle(self) -> None:
        """Main balancing loop"""
        while self.is_active:
            # Balance resources
            allocation = await self.balance()
            
            # Monitor performance
            metrics = await self.monitor_performance()
            
            # Log to audit trail if kernel exists
            if self.kernel and hasattr(self.kernel, 'audit_trail'):
                self.kernel.audit_trail.append({
                    'event': 'aries_balance',
                    'allocation': allocation,
                    'metrics': metrics,
                    'timestamp': time.time()
                })
            
            await asyncio.sleep(30)  # Balance every 30 seconds