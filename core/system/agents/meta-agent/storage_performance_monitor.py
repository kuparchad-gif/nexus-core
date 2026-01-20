# 📂 Path: Systems/engine/memory/storage_performance_monitor.py

import os
import json
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

class StoragePerformanceMonitor:
    """
    Monitors storage performance metrics and generates reports.
    
    This class tracks access times, availability, and usage patterns
    for different storage locations to optimize the hot/warm/cold
    classification.
    """
    
    def __init__(self, log_path: str = "./memory/logs/storage_performance.jsonl"):
        """
        Initialize the storage performance monitor.
        
        Args:
            log_path: Path to the performance log file
        """
        self.log_path = log_path
        self.metrics_cache = {}  # In-memory cache of recent metrics
        self.cache_duration = 3600  # Cache metrics for 1 hour
        self.lock = threading.Lock()
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Create empty log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                pass
    
    def log_operation(self, location_id: str, operation: str, 
                     duration: float, size_bytes: int = 0,
                     success: bool = True) -> None:
        """
        Log a storage operation.
        
        Args:
            location_id: ID of the storage location
            operation: Type of operation (read, write, etc.)
            duration: Duration of the operation in seconds
            size_bytes: Size of data processed in bytes
            success: Whether the operation was successful
        """
        try:
            metric = {
                "timestamp": time.time(),
                "location_id": location_id,
                "operation": operation,
                "duration": duration,
                "size_bytes": size_bytes,
                "success": success,
                "throughput": size_bytes / duration if duration > 0 and size_bytes > 0 else 0
            }
            
            # Add to cache
            with self.lock:
                if location_id not in self.metrics_cache:
                    self.metrics_cache[location_id] = []
                self.metrics_cache[location_id].append(metric)
            
            # Write to log file
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(metric) + "\n")
        except Exception as e:
            print(f"[StoragePerformanceMonitor] Error logging operation: {e}")
    
    def get_location_metrics(self, location_id: str, 
                           time_window: int = 3600) -> Dict[str, Any]:
        """
        Get performance metrics for a specific location.
        
        Args:
            location_id: ID of the storage location
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Calculate cutoff time
            cutoff_time = time.time() - time_window
            
            # Get metrics from cache and log file
            metrics = self._get_metrics_for_location(location_id, cutoff_time)
            
            if not metrics:
                return {
                    "location_id": location_id,
                    "operations_count": 0,
                    "avg_duration": 0,
                    "avg_throughput": 0,
                    "success_rate": 0,
                    "read_count": 0,
                    "write_count": 0,
                    "last_operation_time": None
                }
            
            # Calculate statistics
            total_duration = sum(m["duration"] for m in metrics)
            total_throughput = sum(m["throughput"] for m in metrics if m["throughput"] > 0)
            success_count = sum(1 for m in metrics if m["success"])
            read_count = sum(1 for m in metrics if m["operation"] == "read")
            write_count = sum(1 for m in metrics if m["operation"] == "write")
            
            # Find last operation time
            last_operation = max(metrics, key=lambda m: m["timestamp"])
            
            return {
                "location_id": location_id,
                "operations_count": len(metrics),
                "avg_duration": total_duration / len(metrics) if metrics else 0,
                "avg_throughput": total_throughput / len([m for m in metrics if m["throughput"] > 0]) if any(m["throughput"] > 0 for m in metrics) else 0,
                "success_rate": success_count / len(metrics) if metrics else 0,
                "read_count": read_count,
                "write_count": write_count,
                "last_operation_time": last_operation["timestamp"]
            }
        except Exception as e:
            print(f"[StoragePerformanceMonitor] Error getting metrics: {e}")
            return {
                "location_id": location_id,
                "error": str(e)
            }
    
    def _get_metrics_for_location(self, location_id: str, 
                                cutoff_time: float) -> List[Dict[str, Any]]:
        """
        Get metrics for a location from cache and log file.
        
        Args:
            location_id: ID of the storage location
            cutoff_time: Cutoff timestamp
            
        Returns:
            List of metric dictionaries
        """
        metrics = []
        
        # Get from cache
        with self.lock:
            if location_id in self.metrics_cache:
                metrics.extend([m for m in self.metrics_cache[location_id] 
                               if m["timestamp"] >= cutoff_time])
        
        # If we need to look further back, read from log file
        if cutoff_time < time.time() - self.cache_duration:
            try:
                with open(self.log_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            metric = json.loads(line)
                            if (metric["location_id"] == location_id and 
                                metric["timestamp"] >= cutoff_time and
                                metric["timestamp"] < time.time() - self.cache_duration):
                                metrics.append(metric)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"[StoragePerformanceMonitor] Error reading log file: {e}")
        
        return metrics
    
    def classify_storage_locations(self, location_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Classify storage locations as hot, warm, or cold.
        
        Args:
            location_metrics: Dictionary mapping location_id to metrics
            
        Returns:
            Dictionary mapping location_id to classification
        """
        if not location_metrics:
            return {}
        
        # Extract average durations
        durations = [m["avg_duration"] for m in location_metrics.values() if m["operations_count"] > 0]
        
        if not durations:
            return {loc_id: "unknown" for loc_id in location_metrics}
        
        # Calculate thresholds
        durations.sort()
        hot_threshold = durations[0] * 2  # Twice the fastest
        warm_threshold = durations[0] * 5  # Five times the fastest
        
        # Classify locations
        classifications = {}
        for loc_id, metrics in location_metrics.items():
            if metrics["operations_count"] == 0:
                classifications[loc_id] = "unknown"
            elif metrics["avg_duration"] <= hot_threshold:
                classifications[loc_id] = "hot"
            elif metrics["avg_duration"] <= warm_threshold:
                classifications[loc_id] = "warm"
            else:
                classifications[loc_id] = "cold"
        
        return classifications
    
    def generate_performance_report(self, time_window: int = 86400) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            time_window: Time window in seconds (default: 24 hours)
            
        Returns:
            Dictionary with performance report
        """
        try:
            # Get all unique location IDs
            location_ids = set()
            
            # From cache
            with self.lock:
                location_ids.update(self.metrics_cache.keys())
            
            # From log file
            try:
                with open(self.log_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            metric = json.loads(line)
                            location_ids.add(metric["location_id"])
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"[StoragePerformanceMonitor] Error reading log file: {e}")
            
            # Get metrics for each location
            location_metrics = {}
            for loc_id in location_ids:
                location_metrics[loc_id] = self.get_location_metrics(loc_id, time_window)
            
            # Classify locations
            classifications = self.classify_storage_locations(location_metrics)
            
            # Calculate overall statistics
            total_operations = sum(m["operations_count"] for m in location_metrics.values())
            total_reads = sum(m["read_count"] for m in location_metrics.values())
            total_writes = sum(m["write_count"] for m in location_metrics.values())
            
            # Calculate average success rate
            success_rates = [m["success_rate"] for m in location_metrics.values() if m["operations_count"] > 0]
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
            
            return {
                "timestamp": time.time(),
                "time_window_seconds": time_window,
                "total_operations": total_operations,
                "total_reads": total_reads,
                "total_writes": total_writes,
                "avg_success_rate": avg_success_rate,
                "location_count": len(location_ids),
                "locations": {
                    loc_id: {
                        **metrics,
                        "classification": classifications.get(loc_id, "unknown")
                    }
                    for loc_id, metrics in location_metrics.items()
                }
            }
        except Exception as e:
            print(f"[StoragePerformanceMonitor] Error generating report: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def generate_performance_chart(self, output_path: str = "./memory/reports/storage_performance.png",
                                 time_window: int = 86400) -> bool:
        """
        Generate a performance chart and save it to a file.
        
        Args:
            output_path: Path to save the chart
            time_window: Time window in seconds (default: 24 hours)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get performance report
            report = self.generate_performance_report(time_window)
            
            if "error" in report:
                return False
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot 1: Average Duration by Location
            locations = list(report["locations"].keys())
            durations = [report["locations"][loc]["avg_duration"] * 1000 for loc in locations]  # Convert to ms
            
            # Sort by duration
            sorted_indices = np.argsort(durations)
            sorted_locations = [locations[i] for i in sorted_indices]
            sorted_durations = [durations[i] for i in sorted_indices]
            
            # Color by classification
            colors = []
            for loc in sorted_locations:
                classification = report["locations"][loc]["classification"]
                if classification == "hot":
                    colors.append("green")
                elif classification == "warm":
                    colors.append("orange")
                elif classification == "cold":
                    colors.append("red")
                else:
                    colors.append("gray")
            
            ax1.bar(sorted_locations, sorted_durations, color=colors)
            ax1.set_title("Average Access Duration by Storage Location")
            ax1.set_ylabel("Duration (ms)")
            ax1.set_xlabel("Storage Location")
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Operation Count by Location
            op_counts = [report["locations"][loc]["operations_count"] for loc in sorted_locations]
            read_counts = [report["locations"][loc]["read_count"] for loc in sorted_locations]
            write_counts = [report["locations"][loc]["write_count"] for loc in sorted_locations]
            
            width = 0.35
            ax2.bar(sorted_locations, read_counts, width, label='Reads')
            ax2.bar(sorted_locations, write_counts, width, bottom=read_counts, label='Writes')
            ax2.set_title("Operation Count by Storage Location")
            ax2.set_ylabel("Operation Count")
            ax2.set_xlabel("Storage Location")
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            
            # Add report summary as text
            time_window_hours = time_window / 3600
            plt.figtext(0.5, 0.01, 
                      f"Report Period: {time_window_hours:.1f} hours | "
                      f"Total Operations: {report['total_operations']} | "
                      f"Success Rate: {report['avg_success_rate']*100:.1f}%",
                      ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            plt.savefig(output_path)
            plt.close()
            
            print(f"[StoragePerformanceMonitor] Saved performance chart to {output_path}")
            return True
        except Exception as e:
            print(f"[StoragePerformanceMonitor] Error generating chart: {e}")
            return False
    
    def clean_old_metrics(self, max_age_days: int = 30) -> int:
        """
        Clean old metrics from the log file.
        
        Args:
            max_age_days: Maximum age of metrics to keep in days
            
        Returns:
            Number of records removed
        """
        try:
            cutoff_time = time.time() - (max_age_days * 86400)
            
            # Read all metrics
            metrics = []
            try:
                with open(self.log_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            metric = json.loads(line)
                            if metric["timestamp"] >= cutoff_time:
                                metrics.append(line)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                return 0
            
            # Count removed records
            original_count = sum(1 for _ in open(self.log_path, 'r'))
            removed_count = original_count - len(metrics)
            
            # Write back only recent metrics
            with open(self.log_path, 'w') as f:
                for line in metrics:
                    f.write(line)
            
            print(f"[StoragePerformanceMonitor] Removed {removed_count} old metrics")
            return removed_count
        except Exception as e:
            print(f"[StoragePerformanceMonitor] Error cleaning old metrics: {e}")
            return 0

# 🔥 Example Usage:
if __name__ == "__main__":
    monitor = StoragePerformanceMonitor()
    
    # Simulate some operations
    for i in range(10):
        # Fast storage
        monitor.log_operation("local_hot", "read", 0.01, 1024, True)
        monitor.log_operation("local_hot", "write", 0.02, 2048, True)
        
        # Medium storage
        monitor.log_operation("local_warm", "read", 0.05, 1024, True)
        monitor.log_operation("local_warm", "write", 0.08, 2048, True)
        
        # Slow storage
        monitor.log_operation("local_cold", "read", 0.2, 1024, True)
        monitor.log_operation("local_cold", "write", 0.3, 2048, True)
    
    # Get metrics for a location
    hot_metrics = monitor.get_location_metrics("local_hot")
    print(f"Hot Storage Metrics: {hot_metrics}")
    
    # Generate performance report
    report = monitor.generate_performance_report()
    print(f"Performance Report: {report}")
    
    # Generate performance chart
    monitor.generate_performance_chart()