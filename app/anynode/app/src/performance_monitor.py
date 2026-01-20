#!/usr/bin/env python3
# Performance Monitor for Memory Module

import os
import time
import json
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PerformanceMonitor")

class ExponentialMovingAverage:
    """Exponential moving average for smoothing metrics."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
    
    def update(self, new_value):
        """Update the moving average with a new value."""
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value

class PerformanceMetric:
    """A single performance metric with history."""
    def __init__(self, name, unit="", max_history=1000):
        self.name = name
        self.unit = unit
        self.current = 0.0
        self.min = float('inf')
        self.max = float('-inf')
        self.avg = ExponentialMovingAverage()
        self.history = []
        self.timestamps = []
        self.max_history = max_history
    
    def update(self, value, timestamp=None):
        """Update the metric with a new value."""
        if timestamp is None:
            timestamp = time.time()
        
        self.current = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg.update(value)
        
        # Add to history
        self.history.append(value)
        self.timestamps.append(timestamp)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]
    
    def to_dict(self):
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "unit": self.unit,
            "current": self.current,
            "min": self.min if self.min != float('inf') else None,
            "max": self.max if self.max != float('-inf') else None,
            "avg": self.avg.value,
            "history_length": len(self.history)
        }

class PerformanceMonitor:
    """
    Performance monitoring system for the memory module.
    Tracks metrics, generates reports, and visualizes performance data.
    """
    
    def __init__(self, log_path="./memory/logs/performance.jsonl"):
        """
        Initialize the performance monitor.
        
        Args:
            log_path: Path to the performance log file
        """
        self.log_path = log_path
        self.metrics = {}
        self.lock = threading.Lock()
        self.last_report_time = 0
        self.report_interval = 60  # seconds
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Create empty log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                pass
        
        # Start background monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitor initialized")
    
    def stop(self):
        """Stop the performance monitor."""
        self.running = False
        self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitor stopped")
    
    def _monitor_loop(self):
        """Background monitoring thread."""
        while self.running:
            try:
                current_time = time.time()
                
                # Generate periodic reports
                if current_time - self.last_report_time >= self.report_interval:
                    self.generate_report()
                    self.last_report_time = current_time
                
                # Update system metrics
                self._update_system_metrics()
                
                # Sleep for a bit
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # CPU usage
            cpu_percent = self._get_cpu_percent()
            self.record_metric("system.cpu.percent", cpu_percent, unit="%")
            
            # Memory usage
            memory_info = self._get_memory_info()
            self.record_metric("system.memory.total", memory_info["total"], unit="bytes")
            self.record_metric("system.memory.used", memory_info["used"], unit="bytes")
            self.record_metric("system.memory.percent", memory_info["percent"], unit="%")
            
            # Disk usage
            disk_info = self._get_disk_info()
            self.record_metric("system.disk.total", disk_info["total"], unit="bytes")
            self.record_metric("system.disk.used", disk_info["used"], unit="bytes")
            self.record_metric("system.disk.percent", disk_info["percent"], unit="%")
            
            # GPU metrics if available
            gpu_info = self._get_gpu_info()
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    self.record_metric(f"system.gpu.{i}.memory.used", gpu["memory_used"], unit="bytes")
                    self.record_metric(f"system.gpu.{i}.memory.total", gpu["memory_total"], unit="bytes")
                    self.record_metric(f"system.gpu.{i}.utilization", gpu["utilization"], unit="%")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _get_cpu_percent(self):
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_memory_info(self):
        """Get memory usage information."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "used": mem.used,
                "percent": mem.percent
            }
        except ImportError:
            return {"total": 0, "used": 0, "percent": 0}
    
    def _get_disk_info(self):
        """Get disk usage information."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                "total": disk.total,
                "used": disk.used,
                "percent": disk.percent
            }
        except ImportError:
            return {"total": 0, "used": 0, "percent": 0}
    
    def _get_gpu_info(self):
        """Get GPU usage information if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpus.append({
                    "memory_used": memory.used,
                    "memory_total": memory.total,
                    "utilization": utilization.gpu
                })
            
            pynvml.nvmlShutdown()
            return gpus
        except (ImportError, Exception):
            return []
    
    def record_metric(self, name, value, unit="", timestamp=None):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            timestamp: Optional timestamp (defaults to current time)
        """
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetric(name, unit)
            
            self.metrics[name].update(value, timestamp)
    
    def record_operation_time(self, operation, duration_ms):
        """
        Record the time taken for an operation.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
        """
        self.record_metric(f"operation.{operation}.time", duration_ms, unit="ms")
    
    def record_memory_operation(self, operation, key, size_bytes, duration_ms):
        """
        Record a memory operation.
        
        Args:
            operation: Operation type (e.g., "read", "write")
            key: Memory key
            size_bytes: Size of data in bytes
            duration_ms: Duration in milliseconds
        """
        self.record_metric(f"memory.{operation}.time", duration_ms, unit="ms")
        self.record_metric(f"memory.{operation}.size", size_bytes, unit="bytes")
        
        if size_bytes > 0 and duration_ms > 0:
            throughput = size_bytes / (duration_ms / 1000.0)  # bytes per second
            self.record_metric(f"memory.{operation}.throughput", throughput, unit="bytes/s")
    
    def get_metric(self, name):
        """
        Get a specific metric.
        
        Args:
            name: Metric name
            
        Returns:
            Metric object or None if not found
        """
        with self.lock:
            return self.metrics.get(name)
    
    def get_metrics(self):
        """
        Get all metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self.lock:
            return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    def generate_report(self):
        """
        Generate a performance report and save it to the log file.
        
        Returns:
            Report dictionary
        """
        with self.lock:
            # Create report
            report = {
                "timestamp": time.time(),
                "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()}
            }
            
            # Save to log file
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(report) + "\n")
            except Exception as e:
                logger.error(f"Error saving performance report: {e}")
            
            logger.info(f"Generated performance report with {len(report['metrics'])} metrics")
            return report
    
    def generate_charts(self, output_dir="./memory/reports"):
        """
        Generate performance charts.
        
        Args:
            output_dir: Directory to save charts
            
        Returns:
            List of generated chart paths
        """
        os.makedirs(output_dir, exist_ok=True)
        chart_paths = []
        
        try:
            # Group metrics by category
            categories = {}
            with self.lock:
                for name, metric in self.metrics.items():
                    category = name.split('.')[0] if '.' in name else 'other'
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((name, metric))
            
            # Generate charts for each category
            for category, metrics in categories.items():
                if not metrics:
                    continue
                
                # Create figure
                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                
                # Plot each metric
                for name, metric in metrics:
                    if len(metric.history) < 2:
                        continue
                    
                    # Convert timestamps to relative time (seconds ago)
                    now = time.time()
                    relative_times = [(now - ts) / 60 for ts in metric.timestamps]  # minutes ago
                    
                    # Plot
                    ax.plot(relative_times, metric.history, label=name)
                
                # Set labels and title
                ax.set_xlabel("Minutes Ago")
                ax.set_ylabel("Value")
                ax.set_title(f"{category.capitalize()} Metrics")
                
                # Add legend
                ax.legend()
                
                # Invert x-axis so most recent is on the right
                ax.invert_xaxis()
                
                # Save chart
                chart_path = os.path.join(output_dir, f"{category}_metrics.png")
                plt.savefig(chart_path)
                plt.close(fig)
                
                chart_paths.append(chart_path)
            
            logger.info(f"Generated {len(chart_paths)} performance charts")
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        
        return chart_paths
    
    def get_performance_summary(self):
        """
        Get a summary of performance metrics.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "timestamp": time.time(),
            "system": {},
            "memory": {},
            "operation": {}
        }
        
        with self.lock:
            # System metrics
            for name, metric in self.metrics.items():
                if name.startswith("system."):
                    parts = name.split('.')
                    category = parts[1]
                    if category not in summary["system"]:
                        summary["system"][category] = {}
                    
                    if len(parts) > 2:
                        subcategory = parts[2]
                        if subcategory not in summary["system"][category]:
                            summary["system"][category][subcategory] = {}
                        
                        if len(parts) > 3:
                            summary["system"][category][subcategory][parts[3]] = metric.current
                        else:
                            summary["system"][category][subcategory] = metric.current
                    else:
                        summary["system"][category] = metric.current
            
            # Memory metrics
            for name, metric in self.metrics.items():
                if name.startswith("memory."):
                    parts = name.split('.')
                    if len(parts) >= 3:
                        operation = parts[1]
                        metric_type = parts[2]
                        
                        if operation not in summary["memory"]:
                            summary["memory"][operation] = {}
                        
                        summary["memory"][operation][metric_type] = metric.current
            
            # Operation metrics
            for name, metric in self.metrics.items():
                if name.startswith("operation."):
                    parts = name.split('.')
                    if len(parts) >= 3:
                        operation = parts[1]
                        metric_type = parts[2]
                        
                        if operation not in summary["operation"]:
                            summary["operation"][operation] = {}
                        
                        summary["operation"][operation][metric_type] = metric.current
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Record some test metrics
    for i in range(10):
        monitor.record_metric("test.value", i * 2.5)
        monitor.record_operation_time("test_op", i * 10)
        monitor.record_memory_operation("read", f"key_{i}", 1024 * i, 5 + i)
        time.sleep(0.1)
    
    # Get metrics
    metrics = monitor.get_metrics()
    print(f"Collected {len(metrics)} metrics")
    
    # Generate report
    report = monitor.generate_report()
    print(f"Generated report with {len(report['metrics'])} metrics")
    
    # Generate charts
    charts = monitor.generate_charts()
    print(f"Generated {len(charts)} charts")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"Performance summary: {json.dumps(summary, indent=2)}")
    
    # Stop monitor
    monitor.stop()