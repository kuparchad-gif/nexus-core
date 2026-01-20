#!/usr/bin/env python3
"""
System Optimizer for Viren
Continuously monitors and optimizes system performance
"""

import os
import json
import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemOptimizer")

class SystemOptimizer:
    """Continuously monitors and optimizes system performance"""
    
    def __init__(self):
        """Initialize the system optimizer"""
        self.config_path = os.path.join('C:/Viren/config', 'system_optimizer.json')
        self.stats_path = os.path.join('C:/Viren/logs', 'performance_stats.json')
        self.config = {}
        self.stats = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "components": {}
        }
        self.monitoring = False
        self.monitor_thread = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load optimizer configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default configuration
                self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading optimizer configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default optimizer configuration"""
        self.config = {
            "monitoring": {
                "enabled": True,
                "interval_seconds": 60,
                "history_size": 1440  # 24 hours at 1 minute intervals
            },
            "optimization": {
                "enabled": True,
                "auto_tune": True,
                "cpu_threshold": 80,  # Percentage
                "memory_threshold": 80,  # Percentage
                "disk_threshold": 90,  # Percentage
                "network_threshold": 80  # Percentage
            },
            "components": {
                "weaviate": {
                    "enabled": True,
                    "max_memory_mb": 2048,
                    "optimize_indexes": True
                },
                "binary_protocol": {
                    "enabled": True,
                    "max_shards": 10,
                    "auto_defrag": True
                },
                "intelligence_router": {
                    "enabled": True,
                    "cache_size_mb": 512,
                    "parallel_tasks": 4
                }
            }
        }
        
        self._save_config()
    
    def _save_config(self) -> None:
        """Save optimizer configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimizer configuration: {e}")
    
    def _save_stats(self) -> None:
        """Save performance statistics"""
        try:
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            
            # Trim history if needed
            history_size = self.config["monitoring"]["history_size"]
            for key in ["cpu", "memory", "disk", "network"]:
                if len(self.stats[key]) > history_size:
                    self.stats[key] = self.stats[key][-history_size:]
            
            with open(self.stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance statistics: {e}")
    
    def initialize(self) -> bool:
        """Initialize the system optimizer"""
        logger.info("Initializing System Optimizer")
        
        # Create directories
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        
        # Start monitoring if enabled
        if self.config["monitoring"]["enabled"]:
            self.start_monitoring()
        
        return True
    
    def start_monitoring(self) -> bool:
        """Start performance monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return True
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started performance monitoring")
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop performance monitoring"""
        if not self.monitoring:
            logger.warning("Monitoring not started")
            return True
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped performance monitoring")
        return True
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect system stats
                self._collect_system_stats()
                
                # Collect component stats
                self._collect_component_stats()
                
                # Save stats
                self._save_stats()
                
                # Optimize if needed
                if self.config["optimization"]["enabled"]:
                    self._optimize_system()
                
                # Sleep for the configured interval
                time.sleep(self.config["monitoring"]["interval_seconds"])
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Sleep a bit before retrying
    
    def _collect_system_stats(self) -> None:
        """Collect system performance statistics"""
        timestamp = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.stats["cpu"].append({
            "timestamp": timestamp,
            "percent": cpu_percent
        })
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.stats["memory"].append({
            "timestamp": timestamp,
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        })
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.stats["disk"].append({
            "timestamp": timestamp,
            "total": disk.total,
            "free": disk.free,
            "percent": disk.percent
        })
        
        # Network usage
        net_io = psutil.net_io_counters()
        self.stats["network"].append({
            "timestamp": timestamp,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv
        })
    
    def _collect_component_stats(self) -> None:
        """Collect component-specific statistics"""
        timestamp = time.time()
        
        # In a real system, we would collect stats from each component
        # For now, we'll just use placeholder data
        
        # Weaviate stats
        if "weaviate" not in self.stats["components"]:
            self.stats["components"]["weaviate"] = []
        
        self.stats["components"]["weaviate"].append({
            "timestamp": timestamp,
            "memory_usage_mb": 1024,  # Placeholder
            "query_latency_ms": 50,   # Placeholder
            "index_size_mb": 500      # Placeholder
        })
        
        # Binary Protocol stats
        if "binary_protocol" not in self.stats["components"]:
            self.stats["components"]["binary_protocol"] = []
        
        self.stats["components"]["binary_protocol"].append({
            "timestamp": timestamp,
            "active_shards": 5,       # Placeholder
            "fragmentation": 15,      # Placeholder
            "memory_usage_mb": 300    # Placeholder
        })
        
        # Intelligence Router stats
        if "intelligence_router" not in self.stats["components"]:
            self.stats["components"]["intelligence_router"] = []
        
        self.stats["components"]["intelligence_router"].append({
            "timestamp": timestamp,
            "active_tasks": 2,        # Placeholder
            "cache_usage_mb": 200,    # Placeholder
            "avg_task_time_ms": 150   # Placeholder
        })
    
    def _optimize_system(self) -> None:
        """Optimize system based on collected statistics"""
        # Check if optimization is needed
        needs_optimization = False
        
        # Check CPU usage
        if self.stats["cpu"] and self.stats["cpu"][-1]["percent"] > self.config["optimization"]["cpu_threshold"]:
            needs_optimization = True
            logger.warning(f"CPU usage above threshold: {self.stats['cpu'][-1]['percent']}%")
        
        # Check memory usage
        if self.stats["memory"] and self.stats["memory"][-1]["percent"] > self.config["optimization"]["memory_threshold"]:
            needs_optimization = True
            logger.warning(f"Memory usage above threshold: {self.stats['memory'][-1]['percent']}%")
        
        # Check disk usage
        if self.stats["disk"] and self.stats["disk"][-1]["percent"] > self.config["optimization"]["disk_threshold"]:
            needs_optimization = True
            logger.warning(f"Disk usage above threshold: {self.stats['disk'][-1]['percent']}%")
        
        if needs_optimization:
            logger.info("Performing system optimization")
            self._perform_optimization()
    
    def _perform_optimization(self) -> None:
        """Perform system optimization"""
        # In a real system, we would implement specific optimization strategies
        # For now, we'll just log what we would do
        
        # Optimize Weaviate
        if self.config["components"]["weaviate"]["enabled"]:
            logger.info("Optimizing Weaviate")
            # Example: Reduce memory usage
            if "weaviate" in self.stats["components"] and self.stats["components"]["weaviate"]:
                memory_usage = self.stats["components"]["weaviate"][-1]["memory_usage_mb"]
                if memory_usage > self.config["components"]["weaviate"]["max_memory_mb"]:
                    logger.info(f"Weaviate memory usage ({memory_usage} MB) exceeds limit, would reduce cache size")
        
        # Optimize Binary Protocol
        if self.config["components"]["binary_protocol"]["enabled"]:
            logger.info("Optimizing Binary Protocol")
            # Example: Defragment shards
            if "binary_protocol" in self.stats["components"] and self.stats["components"]["binary_protocol"]:
                fragmentation = self.stats["components"]["binary_protocol"][-1]["fragmentation"]
                if fragmentation > 20 and self.config["components"]["binary_protocol"]["auto_defrag"]:
                    logger.info(f"Binary Protocol fragmentation ({fragmentation}%) is high, would defragment shards")
        
        # Optimize Intelligence Router
        if self.config["components"]["intelligence_router"]["enabled"]:
            logger.info("Optimizing Intelligence Router")
            # Example: Adjust parallel tasks
            if "intelligence_router" in self.stats["components"] and self.stats["components"]["intelligence_router"]:
                active_tasks = self.stats["components"]["intelligence_router"][-1]["active_tasks"]
                if active_tasks > self.config["components"]["intelligence_router"]["parallel_tasks"]:
                    logger.info(f"Intelligence Router has {active_tasks} active tasks, would throttle new tasks")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        # Get the most recent stats
        current_stats = {
            "timestamp": time.time(),
            "cpu": self.stats["cpu"][-1]["percent"] if self.stats["cpu"] else None,
            "memory": self.stats["memory"][-1]["percent"] if self.stats["memory"] else None,
            "disk": self.stats["disk"][-1]["percent"] if self.stats["disk"] else None,
            "components": {}
        }
        
        # Add component stats
        for component, stats in self.stats["components"].items():
            if stats:
                current_stats["components"][component] = stats[-1]
        
        return current_stats
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status"""
        return {
            "enabled": self.config["optimization"]["enabled"],
            "auto_tune": self.config["optimization"]["auto_tune"],
            "thresholds": {
                "cpu": self.config["optimization"]["cpu_threshold"],
                "memory": self.config["optimization"]["memory_threshold"],
                "disk": self.config["optimization"]["disk_threshold"],
                "network": self.config["optimization"]["network_threshold"]
            },
            "components": {
                component: config["enabled"]
                for component, config in self.config["components"].items()
            }
        }
    
    def set_optimization_config(self, config: Dict[str, Any]) -> bool:
        """Update optimization configuration"""
        try:
            # Update config
            if "enabled" in config:
                self.config["optimization"]["enabled"] = bool(config["enabled"])
            
            if "auto_tune" in config:
                self.config["optimization"]["auto_tune"] = bool(config["auto_tune"])
            
            if "thresholds" in config:
                for key, value in config["thresholds"].items():
                    if key in self.config["optimization"]:
                        self.config["optimization"][f"{key}_threshold"] = int(value)
            
            if "components" in config:
                for component, enabled in config["components"].items():
                    if component in self.config["components"]:
                        self.config["components"][component]["enabled"] = bool(enabled)
            
            # Save config
            self._save_config()
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating optimization configuration: {e}")
            return False