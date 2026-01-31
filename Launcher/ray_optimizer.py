"""
RAY Distributed Computing Optimizer
Provides distributed task execution and parallel processing
"""

import ray
from ray import serve
from ray.util.multiprocessing import Pool as RayPool
from typing import List, Dict, Any, Callable, Optional
import logging
import numpy as np
from functools import wraps
import time

logger = logging.getLogger(__name__)


class RAYOptimizer:
    """
    RAY-based distributed computing and parallel processing
    - Task parallelization
    - Actor-based stateful computations
    - Resource management
    - GPU acceleration support
    """
    
    def __init__(self, num_cpus: Optional[int] = None, 
                 num_gpus: Optional[int] = 0,
                 memory: Optional[int] = None):
        """
        Initialize RAY cluster
        num_cpus: Number of CPUs to use (None = all available)
        num_gpus: Number of GPUs to use
        memory: Memory limit in bytes
        """
        self.initialized = False
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        
        try:
            # Check if Ray is already initialized
            if not ray.is_initialized():
                ray.init(
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    _memory=memory,
                    ignore_reinit_error=True,
                    logging_level=logging.INFO
                )
            self.initialized = True
            logger.info(f"RAY initialized: {ray.cluster_resources()}")
        except Exception as e:
            logger.error(f"RAY initialization failed: {e}")
            raise
    
    def get_cluster_resources(self) -> Dict[str, Any]:
        """Get available cluster resources"""
        if not self.initialized:
            return {}
        
        return {
            "total": ray.cluster_resources(),
            "available": ray.available_resources(),
            "nodes": len(ray.nodes())
        }
    
    # ==================== TASK PARALLELIZATION ====================
    
    @staticmethod
    def parallelize(func: Callable) -> Callable:
        """
        Decorator to make function executable in parallel with RAY
        Usage: @RAYOptimizer.parallelize
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            @ray.remote
            def remote_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            return remote_func.remote(*args, **kwargs)
        
        return wrapper
    
    def parallel_map(self, func: Callable, items: List[Any], 
                     num_workers: Optional[int] = None) -> List[Any]:
        """
        Execute function in parallel across items
        Returns results in same order as input
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote
        def remote_func(item):
            return func(item)
        
        # Submit all tasks
        futures = [remote_func.remote(item) for item in items]
        
        # Gather results
        results = ray.get(futures)
        
        return results
    
    def parallel_batch_map(self, func: Callable, items: List[Any],
                           batch_size: int = 10) -> List[Any]:
        """
        Execute function in parallel with batching for efficiency
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote
        def batch_func(batch):
            return [func(item) for item in batch]
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Submit batch tasks
        futures = [batch_func.remote(batch) for batch in batches]
        
        # Gather and flatten results
        batch_results = ray.get(futures)
        results = [item for batch in batch_results for item in batch]
        
        return results
    
    def parallel_starmap(self, func: Callable, args_list: List[tuple]) -> List[Any]:
        """
        Execute function with multiple arguments in parallel
        args_list: [(arg1, arg2, ...), (arg1, arg2, ...), ...]
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote
        def remote_func(*args):
            return func(*args)
        
        futures = [remote_func.remote(*args) for args in args_list]
        results = ray.get(futures)
        
        return results
    
    # ==================== ACTOR-BASED COMPUTATIONS ====================
    
    def create_actor(self, actor_class: type, *args, **kwargs):
        """
        Create RAY actor for stateful computations
        Returns actor handle
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        RemoteActor = ray.remote(actor_class)
        actor = RemoteActor.remote(*args, **kwargs)
        
        return actor
    
    def create_actor_pool(self, actor_class: type, num_actors: int,
                          *args, **kwargs) -> List:
        """Create pool of actors for load balancing"""
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        RemoteActor = ray.remote(actor_class)
        actors = [RemoteActor.remote(*args, **kwargs) for _ in range(num_actors)]
        
        return actors
    
    # ==================== RESOURCE MANAGEMENT ====================
    
    def execute_with_resources(self, func: Callable, 
                               num_cpus: float = 1.0,
                               num_gpus: float = 0.0,
                               memory: Optional[int] = None,
                               *args, **kwargs) -> Any:
        """
        Execute function with specific resource requirements
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus, memory=memory)
        def resource_func(*args, **kwargs):
            return func(*args, **kwargs)
        
        future = resource_func.remote(*args, **kwargs)
        result = ray.get(future)
        
        return result
    
    # ==================== ADVANCED PATTERNS ====================
    
    def pipeline(self, stages: List[Callable], data: Any) -> Any:
        """
        Execute pipeline of functions in sequence with RAY
        Each stage processes output of previous stage
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote
        def stage_func(func, input_data):
            return func(input_data)
        
        current_data = data
        for stage in stages:
            future = stage_func.remote(stage, current_data)
            current_data = ray.get(future)
        
        return current_data
    
    def parallel_pipeline(self, stages: List[Callable], data_items: List[Any]) -> List[Any]:
        """
        Execute pipeline on multiple data items in parallel
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote
        def pipeline_func(data):
            result = data
            for stage in stages:
                result = stage(result)
            return result
        
        futures = [pipeline_func.remote(item) for item in data_items]
        results = ray.get(futures)
        
        return results
    
    def map_reduce(self, map_func: Callable, reduce_func: Callable,
                   data: List[Any], num_partitions: Optional[int] = None) -> Any:
        """
        Distributed map-reduce pattern
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        if num_partitions is None:
            num_partitions = len(ray.cluster_resources().get('CPU', 1))
        
        # Partition data
        partition_size = len(data) // num_partitions
        partitions = [data[i:i + partition_size] 
                     for i in range(0, len(data), partition_size)]
        
        @ray.remote
        def map_partition(partition):
            return [map_func(item) for item in partition]
        
        @ray.remote
        def reduce_partition(results):
            return reduce_func(results)
        
        # Map phase
        map_futures = [map_partition.remote(part) for part in partitions]
        map_results = ray.get(map_futures)
        
        # Flatten results
        all_mapped = [item for partition in map_results for item in partition]
        
        # Reduce phase
        reduce_future = reduce_partition.remote(all_mapped)
        final_result = ray.get(reduce_future)
        
        return final_result
    
    # ==================== OPTIMIZATION UTILITIES ====================
    
    def optimize_batch_size(self, func: Callable, data: List[Any],
                           test_sizes: List[int] = [1, 10, 50, 100]) -> int:
        """
        Automatically determine optimal batch size for function
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        test_data = data[:min(1000, len(data))]
        best_size = 1
        best_time = float('inf')
        
        for batch_size in test_sizes:
            start_time = time.time()
            try:
                self.parallel_batch_map(func, test_data, batch_size=batch_size)
                elapsed = time.time() - start_time
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_size = batch_size
            except Exception as e:
                logger.warning(f"Batch size {batch_size} failed: {e}")
        
        logger.info(f"Optimal batch size: {best_size} (time: {best_time:.2f}s)")
        return best_size
    
    def adaptive_parallel_map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Automatically optimize and execute parallel map
        """
        if len(items) < 10:
            # Too few items, just run sequentially
            return [func(item) for item in items]
        
        # Determine optimal batch size
        optimal_batch = self.optimize_batch_size(func, items)
        
        # Execute with optimal batch size
        return self.parallel_batch_map(func, items, batch_size=optimal_batch)
    
    # ==================== GPU ACCELERATION ====================
    
    def gpu_map(self, func: Callable, items: List[Any],
                gpus_per_task: float = 0.25) -> List[Any]:
        """
        Execute function on GPU in parallel
        gpus_per_task: Fraction of GPU to allocate per task
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        @ray.remote(num_gpus=gpus_per_task)
        def gpu_func(item):
            return func(item)
        
        futures = [gpu_func.remote(item) for item in items]
        results = ray.get(futures)
        
        return results
    
    # ==================== MONITORING & DEBUGGING ====================
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about running tasks"""
        if not self.initialized:
            return {}
        
        return {
            "resources": self.get_cluster_resources(),
            "nodes": ray.nodes(),
            "timeline": ray.timeline()
        }
    
    def wait_for_tasks(self, futures: List, num_returns: int = 1,
                       timeout: Optional[float] = None) -> tuple:
        """
        Wait for specified number of tasks to complete
        Returns (ready_futures, remaining_futures)
        """
        if not self.initialized:
            raise RuntimeError("RAY not initialized")
        
        ready, remaining = ray.wait(futures, num_returns=num_returns, timeout=timeout)
        return ready, remaining
    
    # ==================== CLEANUP ====================
    
    def shutdown(self):
        """Shutdown RAY cluster"""
        if self.initialized:
            ray.shutdown()
            self.initialized = False
            logger.info("RAY shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()


# ==================== HELPER ACTORS ====================

@ray.remote
class WorkerActor:
    """Generic worker actor for stateful computations"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.task_count = 0
        self.state = {}
    
    def process(self, func: Callable, data: Any) -> Any:
        """Process data with function"""
        self.task_count += 1
        return func(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "task_count": self.task_count,
            "state_size": len(self.state)
        }
    
    def update_state(self, key: str, value: Any):
        """Update worker state"""
        self.state[key] = value
    
    def get_state(self, key: str) -> Any:
        """Get state value"""
        return self.state.get(key)


@ray.remote
class DistributedCache:
    """Distributed cache actor for sharing data across tasks"""
    
    def __init__(self):
        self.cache = {}
    
    def put(self, key: str, value: Any):
        """Store value in cache"""
        self.cache[key] = value
    
    def get(self, key: str) -> Any:
        """Retrieve value from cache"""
        return self.cache.get(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.cache
    
    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


# Global instance
ray_optimizer = None

def get_ray_optimizer(num_cpus: Optional[int] = None, 
                      num_gpus: int = 0) -> RAYOptimizer:
    """Get or create global RAY optimizer instance"""
    global ray_optimizer
    
    if ray_optimizer is None:
        ray_optimizer = RAYOptimizer(num_cpus=num_cpus, num_gpus=num_gpus)
    
    return ray_optimizer
