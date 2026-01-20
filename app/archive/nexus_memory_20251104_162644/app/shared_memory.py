#!/usr/bin/env python3
# Shared Memory Implementation for Nexus Memory Module

import os
import mmap
import ctypes
import threading
import time
from typing import Dict, Any, Optional, List

# Constants
SHARED_MEMORY_SIZE  =  1024 * 1024 * 1024  # 1GB default
QUEUE_SIZE  =  10000

class AtomicCounter:
    """Thread-safe counter implementation."""
    def __init__(self, initial_value = 0):
        self._value  =  initial_value
        self._lock  =  threading.Lock()

    @property
    def value(self):
        with self._lock:
            return self._value

    def increment(self):
        with self._lock:
            self._value + =  1
            return self._value

    def decrement(self):
        with self._lock:
            self._value - =  1
            return self._value

class SharedMemoryContext:
    """Context for managing shared memory access."""
    def __init__(self, shared_ptr, size):
        self.ptr  =  shared_ptr
        self.size  =  size
        self.allocated  =  0
        self.lock  =  threading.Lock()

    def allocate(self, size):
        """Allocate a block of shared memory."""
        with self.lock:
            if self.allocated + size > self.size:
                return None  # Out of memory

            block_ptr  =  self.ptr + self.allocated
            self.allocated + =  size
            return block_ptr

class SharedRingBuffer:
    """Lock-free ring buffer for shared memory communication."""
    def __init__(self, memory_ptr, buffer_size):
        self.buffer  =  memory_ptr
        self.size  =  buffer_size
        self.write_idx  =  AtomicCounter(0)
        self.read_idx  =  AtomicCounter(0)

    def write(self, data):
        """Write data to the ring buffer."""
        if self.depth() > =  self.size:
            return None  # Buffer full

        idx  =  self.write_idx.increment() % self.size
        self.buffer[idx]  =  data
        return idx

    def read(self):
        """Read data from the ring buffer."""
        if self.read_idx.value == self.write_idx.value:
            return None  # Buffer empty

        idx  =  self.read_idx.increment() % self.size
        return self.buffer[idx]

    def depth(self):
        """Get current buffer depth."""
        return self.write_idx.value - self.read_idx.value

def initialize_shared_memory(name = "/nexus_shared_memory", size = None):
    """Initialize shared memory region."""
    if size is None:
        # Try to get size from environment
        size_str  =  os.environ.get("SHARED_MEMORY_SIZE", "1G")
        if size_str.endswith("G"):
            size  =  int(size_str[:-1]) * 1024 * 1024 * 1024
        elif size_str.endswith("M"):
            size  =  int(size_str[:-1]) * 1024 * 1024
        else:
            size  =  SHARED_MEMORY_SIZE

    try:
        # For Linux/Unix systems
        fd  =  os.open(name, os.O_CREAT | os.O_RDWR)
        os.ftruncate(fd, size)
        shared_ptr  =  mmap.mmap(fd, size, mmap.MAP_SHARED)
        os.close(fd)
    except (AttributeError, OSError):
        # For Windows systems
        try:
            shared_ptr  =  mmap.mmap(-1, size, tagname = name)
        except Exception as e:
            print(f"Failed to initialize shared memory: {e}")
            return None

    print(f"Initialized shared memory of size {size} bytes")
    return SharedMemoryContext(shared_ptr, size)

class SharedMemoryManager:
    """Manager for shared memory operations."""
    def __init__(self, name = "/nexus_shared_memory", size = None):
        self.context  =  initialize_shared_memory(name, size)
        if not self.context:
            raise RuntimeError("Failed to initialize shared memory")

        # Create ring buffer for task queue
        buffer_size  =  min(QUEUE_SIZE, self.context.size // 1024)  # Ensure buffer fits in shared memory
        buffer_ptr  =  self.context.allocate(buffer_size * 8)  # 8 bytes per entry
        self.task_queue  =  SharedRingBuffer(buffer_ptr, buffer_size)

        # Create data region
        data_size  =  self.context.size - buffer_size * 8
        self.data_ptr  =  self.context.allocate(data_size)
        self.data_size  =  data_size
        self.allocated_blocks  =  {}
        self.free_blocks  =  [(0, data_size)]
        self.block_lock  =  threading.Lock()

    def allocate_block(self, size):
        """Allocate a block in the data region."""
        with self.block_lock:
            # Find a free block that's large enough
            for i, (start, block_size) in enumerate(self.free_blocks):
                if block_size > =  size:
                    # Found a suitable block
                    self.free_blocks.pop(i)

                    # If block is larger than needed, split it
                    if block_size > size:
                        self.free_blocks.append((start + size, block_size - size))

                    # Allocate the block
                    block_id  =  id(object())  # Generate unique ID
                    self.allocated_blocks[block_id]  =  (start, size)
                    return block_id, self.data_ptr + start

            return None, None  # No suitable block found

    def free_block(self, block_id):
        """Free an allocated block."""
        with self.block_lock:
            if block_id not in self.allocated_blocks:
                return False

            start, size  =  self.allocated_blocks.pop(block_id)
            self.free_blocks.append((start, size))

            # Merge adjacent free blocks
            self.free_blocks.sort()
            i  =  0
            while i < len(self.free_blocks) - 1:
                curr_start, curr_size  =  self.free_blocks[i]
                next_start, next_size  =  self.free_blocks[i + 1]

                if curr_start + curr_size == next_start:
                    # Merge blocks
                    self.free_blocks[i]  =  (curr_start, curr_size + next_size)
                    self.free_blocks.pop(i + 1)
                else:
                    i + =  1

            return True

    def write_data(self, data, block_id = None):
        """Write data to shared memory."""
        if block_id is None:
            # Allocate new block
            block_id, ptr  =  self.allocate_block(len(data))
            if block_id is None:
                return None
        else:
            # Use existing block
            if block_id not in self.allocated_blocks:
                return None

            start, size  =  self.allocated_blocks[block_id]
            if size < len(data):
                return None

            ptr  =  self.data_ptr + start

        # Write data
        ptr[:len(data)]  =  data
        return block_id

    def read_data(self, block_id):
        """Read data from shared memory."""
        if block_id not in self.allocated_blocks:
            return None

        start, size  =  self.allocated_blocks[block_id]
        ptr  =  self.data_ptr + start
        return ptr[:size]

    def submit_task(self, task_data):
        """Submit a task to the queue."""
        # Write task data to shared memory
        block_id  =  self.write_data(task_data)
        if block_id is None:
            return False

        # Add to task queue
        if self.task_queue.write(block_id) is None:
            # Queue full, free the block
            self.free_block(block_id)
            return False

        return True

    def get_task(self):
        """Get a task from the queue."""
        block_id  =  self.task_queue.read()
        if block_id is None:
            return None

        data  =  self.read_data(block_id)
        self.free_block(block_id)
        return data

# Example usage
if __name__ == "__main__":
    # Initialize shared memory
    manager  =  SharedMemoryManager()

    # Producer thread
    def producer():
        for i in range(100):
            data  =  f"Task {i}".encode()
            success  =  manager.submit_task(data)
            print(f"Produced task {i}: {'success' if success else 'failed'}")
            time.sleep(0.01)

    # Consumer thread
    def consumer():
        for _ in range(100):
            task  =  manager.get_task()
            if task:
                print(f"Consumed task: {task.decode()}")
            time.sleep(0.02)

    # Start threads
    threading.Thread(target = producer).start()
    threading.Thread(target = consumer).start()