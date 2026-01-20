#!/usr/bin/env python3
# Memory Module - Main Integration Point

import os
import time
import threading
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoryModule")

# Import local modules
from .shared_memory import SharedMemoryManager
from .binary_emotion import encode_emotion, decode_emotion, encode_emotional_vector, decode_emotional_vector
from .cuda_processor import EmotionalProcessor, pin_cpu_to_cuda
from .smart_switch import SharedMemorySwitch
from .binary_protocol import BinaryProtocol
from .performance_monitor import PerformanceMonitor

# Try to import LLM modules
try:
    from ..llm.llm_manager import LLMManager
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("LLM modules not available. Running without LLM support.")
    LLM_AVAILABLE = False

class MemoryModule:
    """
    Main Memory Module that integrates all components.
    """
    
    def __init__(self, config_path="./Config/memory_config.json"):
        """
        Initialize the Memory Module.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.shared_memory = SharedMemoryManager()
        self.performance_monitor = PerformanceMonitor()
        self.emotional_processor = EmotionalProcessor(self.shared_memory)
        self.smart_switch = SharedMemorySwitch(self.shared_memory)
        self.binary_protocol = BinaryProtocol()
        
        # Initialize LLMs if available
        self.llm_manager = None
        if LLM_AVAILABLE:
            self.llm_manager = LLMManager()
        
        # State
        self.running = False
        self.services = {}
        
        logger.info("Memory Module initialized")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found. Using defaults.")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def start(self):
        """Start the Memory Module and all its components."""
        if self.running:
            logger.warning("Memory Module already running")
            return
        
        logger.info("Starting Memory Module...")
        
        # Start components
        self.emotional_processor.start()
        self.smart_switch.start()
        
        # Start services
        self._start_services()
        
        self.running = True
        logger.info("Memory Module started")
    
    def stop(self):
        """Stop the Memory Module and all its components."""
        if not self.running:
            logger.warning("Memory Module not running")
            return
        
        logger.info("Stopping Memory Module...")
        
        # Stop services
        self._stop_services()
        
        # Stop components
        self.smart_switch.stop()
        self.emotional_processor.stop()
        self.performance_monitor.stop()
        
        self.running = False
        logger.info("Memory Module stopped")
    
    def _start_services(self):
        """Start all services."""
        # Import services dynamically to avoid circular imports
        from .memory_service import MemoryService
        from .archive_service import ArchiveService
        from .planner_service import PlannerService
        
        # Create services
        self.services["memory"] = MemoryService(
            shared_memory=self.shared_memory,
            performance_monitor=self.performance_monitor,
            llm_manager=self.llm_manager
        )
        
        self.services["archive"] = ArchiveService(
            shared_memory=self.shared_memory,
            performance_monitor=self.performance_monitor,
            llm_manager=self.llm_manager
        )
        
        self.services["planner"] = PlannerService(
            shared_memory=self.shared_memory,
            performance_monitor=self.performance_monitor,
            llm_manager=self.llm_manager
        )
        
        # Start services
        for name, service in self.services.items():
            service.start()
            logger.info(f"Started {name} service")
    
    def _stop_services(self):
        """Stop all services."""
        for name, service in self.services.items():
            service.stop()
            logger.info(f"Stopped {name} service")
    
    def get_status(self):
        """Get status of the Memory Module and its components."""
        status = {
            "running": self.running,
            "timestamp": time.time(),
            "services": {},
            "components": {
                "emotional_processor": {
                    "devices": self.emotional_processor.get_device_stats() if self.running else []
                },
                "smart_switch": self.smart_switch.get_stats() if self.running else {},
                "performance": self.performance_monitor.get_performance_summary() if self.running else {}
            }
        }
        
        # Add service status
        for name, service in self.services.items():
            status["services"][name] = service.get_status() if self.running else {"running": False}
        
        return status

def setup_cpu_pinning():
    """Set up CPU pinning based on environment variables."""
    try:
        # Get CPU affinity from environment
        cpu_affinity = os.environ.get("CPU_AFFINITY", "")
        if not cpu_affinity:
            logger.info("No CPU affinity specified")
            return
        
        # Parse CPU IDs
        cpu_ids = []
        for part in cpu_affinity.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpu_ids.extend(range(start, end + 1))
            else:
                cpu_ids.append(int(part))
        
        # Get CUDA device
        cuda_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        
        # Pin CPUs
        for cpu_id in cpu_ids:
            pin_cpu_to_cuda(cpu_id, cuda_device)
            logger.info(f"Pinned CPU {cpu_id} to CUDA device {cuda_device}")
    except Exception as e:
        logger.error(f"Error setting up CPU pinning: {e}")

def boot_memory_module():
    """Boot the Memory Module."""
    logger.info("Booting Memory Module...")
    
    # Set up CPU pinning
    setup_cpu_pinning()
    
    # Initialize and start Memory Module
    module = MemoryModule()
    module.start()
    
    logger.info("Memory Module boot complete")
    return module

# Main entry point
if __name__ == "__main__":
    # Boot Memory Module
    memory_module = boot_memory_module()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop Memory Module
        memory_module.stop()
        logger.info("Memory Module shutdown complete")