#!/usr/bin/env python3
# Smart Switch for Dynamic Task Distribution

import os
import time
import threading
import queue
import json
from typing import Dict, List, Any, Optional, Tuple, Union

# Import local modules
from .shared_memory import SharedMemoryManager
from .cuda_processor import CudaDevice

class Task:
    """Represents a processing task."""
    def __init__(self, task_id, task_type, data, priority=3, source=None):
        self.id = task_id
        self.type = task_type
        self.data = data
        self.priority = priority
        self.source = source
        self.timestamp = time.time()
        self.status = "pending"
        self.result = None
        self.processing_time = None
    
    def to_dict(self):
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority,
            "source": self.source,
            "timestamp": self.timestamp,
            "status": self.status,
            "processing_time": self.processing_time
        }

class DeviceManager:
    """Manages a collection of processing devices."""
    def __init__(self):
        self.devices = []
        self.device_lock = threading.Lock()
        
        # Try to detect CUDA devices
        self._detect_cuda_devices()
        
        # If no devices found, add CPU fallback
        if not self.devices:
            self.devices.append(CudaDevice(-1))  # -1 indicates CPU
    
    def _detect_cuda_devices(self):
        """Detect available CUDA devices."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            device_count = cuda.Device.count()
            
            for i in range(device_count):
                self.devices.append(CudaDevice(i))
            
            print(f"Detected {len(self.devices)} CUDA devices")
        except ImportError:
            print("CUDA not available")
        except Exception as e:
            print(f"Error detecting CUDA devices: {e}")
    
    def get_device_count(self):
        """Get number of available devices."""
        return len(self.devices)
    
    def get_device(self, device_id):
        """Get a specific device by ID."""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_devices(self):
        """Get all available devices."""
        return self.devices
    
    def get_device_stats(self):
        """Get statistics for all devices."""
        return [device.get_stats() for device in self.devices]

class SmartSwitch:
    """
    Smart switch for dynamic task distribution across processing devices.
    """
    def __init__(self, shared_memory=None):
        self.device_manager = DeviceManager()
        self.shared_memory = shared_memory
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.tasks = {}  # task_id -> Task
        self.running = False
        self.workers = []
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "queue_depth": 0
        }
        self.stats_lock = threading.Lock()
    
    def start(self, num_workers=None):
        """Start the smart switch."""
        if self.running:
            return
        
        self.running = True
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, self.device_manager.get_device_count())
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Start result collector thread
        self.result_collector = threading.Thread(target=self._result_collector_loop, daemon=True)
        self.result_collector.start()
        
        print(f"Smart switch started with {num_workers} workers")
    
    def stop(self):
        """Stop the smart switch."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        # Wait for result collector
        self.result_collector.join(timeout=1.0)
        
        print("Smart switch stopped")
    
    def submit_task(self, task_type, data, priority=3, source=None):
        """
        Submit a task for processing.
        
        Args:
            task_type: Type of task
            data: Task data
            priority: Priority (lower number = higher priority)
            source: Source of the task
            
        Returns:
            Task ID if successful, None otherwise
        """
        # Create task
        task_id = f"task-{int(time.time() * 1000)}-{self.stats['tasks_submitted']}"
        task = Task(task_id, task_type, data, priority, source)
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to queue
        self.task_queue.put((priority, task_id))
        
        # Update stats
        with self.stats_lock:
            self.stats["tasks_submitted"] += 1
            self.stats["queue_depth"] = self.task_queue.qsize()
        
        return task_id
    
    def get_task_status(self, task_id):
        """Get status of a specific task."""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None
    
    def get_completed_tasks(self, limit=10):
        """Get completed tasks."""
        completed = []
        try:
            for _ in range(limit):
                task_id = self.result_queue.get_nowait()
                if task_id in self.tasks:
                    completed.append(self.tasks[task_id])
                self.result_queue.task_done()
        except queue.Empty:
            pass
        
        return completed
    
    def get_stats(self):
        """Get smart switch statistics."""
        with self.stats_lock:
            stats_copy = self.stats.copy()
        
        # Add device stats
        stats_copy["devices"] = self.device_manager.get_device_stats()
        
        return stats_copy
    
    def _worker_loop(self, worker_id):
        """Worker thread loop."""
        print(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                try:
                    priority, task_id = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Get task
                if task_id not in self.tasks:
                    self.task_queue.task_done()
                    continue
                
                task = self.tasks[task_id]
                task.status = "processing"
                
                # Select device
                device = self._select_device(task)
                
                # Process task
                start_time = time.time()
                success = self._process_task(task, device)
                task.processing_time = time.time() - start_time
                
                # Update task status
                task.status = "completed" if success else "failed"
                
                # Update stats
                with self.stats_lock:
                    if success:
                        self.stats["tasks_completed"] += 1
                    else:
                        self.stats["tasks_failed"] += 1
                    
                    # Update average processing time
                    if self.stats["avg_processing_time"] == 0.0:
                        self.stats["avg_processing_time"] = task.processing_time
                    else:
                        self.stats["avg_processing_time"] = (
                            0.9 * self.stats["avg_processing_time"] + 
                            0.1 * task.processing_time
                        )
                    
                    self.stats["queue_depth"] = self.task_queue.qsize()
                
                # Add to result queue if successful
                if success:
                    self.result_queue.put(task_id)
                
                # Mark task as done
                self.task_queue.task_done()
            
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
                time.sleep(0.1)  # Avoid tight loop on error
    
    def _result_collector_loop(self):
        """Result collector thread loop."""
        print("Result collector started")
        
        while self.running:
            try:
                # Sleep a bit
                time.sleep(0.1)
                
                # Clean up old tasks
                self._cleanup_old_tasks()
            except Exception as e:
                print(f"Error in result collector: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks."""
        current_time = time.time()
        to_remove = []
        
        for task_id, task in self.tasks.items():
            # Keep tasks for 5 minutes after completion
            if task.status in ["completed", "failed"] and current_time - task.timestamp > 300:
                to_remove.append(task_id)
        
        # Remove old tasks
        for task_id in to_remove:
            del self.tasks[task_id]
    
    def _select_device(self, task):
        """
        Select the best device for a task.
        
        Args:
            task: Task to process
            
        Returns:
            Selected device
        """
        devices = self.device_manager.get_devices()
        if not devices:
            return None
        
        # Calculate priority score for each device
        scores = []
        for device in devices:
            # Base score is inverse of current load
            score = 1.0 / (1.0 + device.get_load())
            
            # Adjust for memory requirements
            memory_required = len(task.data) if task.data else 0
            if hasattr(device, 'available_memory') and device.available_memory > 0:
                if memory_required <= device.available_memory:
                    score *= 2.0
                else:
                    score *= 0.1  # Heavy penalty if memory insufficient
            
            scores.append((device, score))
        
        # Return device with highest score
        return max(scores, key=lambda x: x[1])[0]
    
    def _process_task(self, task, device):
        """
        Process a task on a device.
        
        Args:
            task: Task to process
            device: Device to use
            
        Returns:
            True if successful, False otherwise
        """
        if not device:
            return False
        
        try:
            # Submit task to device
            return device.submit_task(task)
        except Exception as e:
            print(f"Error processing task {task.id}: {e}")
            return False

class SharedMemorySwitch(SmartSwitch):
    """
    Smart switch that uses shared memory for task distribution.
    """
    def __init__(self, shared_memory):
        super().__init__(shared_memory)
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
    
    def start(self, num_workers=None):
        """Start the shared memory switch."""
        super().start(num_workers)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop the shared memory switch."""
        super().stop()
        self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Monitor shared memory for tasks."""
        print("Shared memory monitor started")
        
        while self.running:
            try:
                # Get task from shared memory
                task_data = self.shared_memory.get_task()
                if not task_data:
                    time.sleep(0.001)  # Short sleep if no tasks
                    continue
                
                # Parse task
                task_type = task_data[0] if task_data else 0
                
                # Submit task
                self.submit_task(task_type, task_data, priority=3, source="shared_memory")
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(0.1)  # Avoid tight loop on error
    
    def notify_new_task(self):
        """Notify that a new task is available in shared memory."""
        # This is a no-op since we're constantly monitoring shared memory
        pass

# Example usage
if __name__ == "__main__":
    from .shared_memory import SharedMemoryManager
    
    # Initialize shared memory
    shared_memory = SharedMemoryManager()
    
    # Initialize smart switch
    switch = SharedMemorySwitch(shared_memory)
    switch.start()
    
    # Submit some test tasks
    for i in range(10):
        task_data = f"Task {i}".encode()
        task_id = switch.submit_task(1, task_data, priority=3, source="test")
        print(f"Submitted task {task_id}")
    
    # Wait for processing
    time.sleep(2.0)
    
    # Get completed tasks
    completed = switch.get_completed_tasks()
    for task in completed:
        print(f"Completed task {task.id} in {task.processing_time:.3f}s")
    
    # Get stats
    stats = switch.get_stats()
    print(f"Stats: {stats}")
    
    # Stop switch
    switch.stop()