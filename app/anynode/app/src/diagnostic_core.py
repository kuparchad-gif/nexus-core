#!/usr/bin/env python3
"""
Cloud Viren Diagnostic Core
Core diagnostic engine with Gemma 3 3B integration
"""

import os
import sys
import json
import time
import logging
import platform
import subprocess
import threading
import psutil
import requests
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("viren_diagnostic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VirenDiagnostic")

# Company theme colors
THEME_COLORS = {
    "plumb": "#A2799A",    # Rich purple
    "primer": "#93AEC5",   # Medium blue
    "silver": "#AFC5DC",   # Light blue
    "putty": "#C6D6E2",    # Very light blue
    "dried_putty": "#D8E3EB",  # Pale blue
    "white": "#EBF2F6"     # Off-white
}

class DiagnosticCore:
    """
    Core diagnostic engine for Cloud Viren
    Integrates with Gemma 3 3B for intelligent analysis
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the diagnostic core"""
        self.config_path = config_path or os.path.join("config", "diagnostic_config.json")
        self.config = self._load_config()
        self.system_info = self._get_system_info()
        self.diagnostic_modules = {}
        self.active_diagnostics = {}
        self.diagnostic_history = []
        self.max_history = 100
        self.llm_client = None
        self.research_tentacles = None
        self.blockchain_relay = None
        self.cloud_sync_status = {"last_sync": None, "status": "disconnected"}
        
        # Initialize components
        self._init_diagnostic_modules()
        self._init_llm_client()
        self._init_research_tentacles()
        self._init_blockchain_relay()
        
        logger.info(f"Diagnostic Core initialized on {self.system_info['hostname']} ({self.system_info['os']})")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "llm_model": "gemma-3-3b",
            "llm_endpoint": "http://localhost:11434/api/generate",
            "cloud_endpoint": "https://api.viren-cloud.com/v1",
            "api_key": "",
            "diagnostic_interval": 3600,  # 1 hour
            "blockchain_relay": {
                "enabled": True,
                "node_endpoint": "https://relay.nexus-blockchain.io",
                "idle_threshold": 1800  # 30 minutes
            },
            "research_tentacles": {
                "enabled": True,
                "max_tentacles": 5,
                "search_timeout": 30
            },
            "monitoring": {
                "cpu": True,
                "memory": True,
                "disk": True,
                "network": True,
                "processes": True,
                "services": True
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config.get(key), dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    
                    logger.info("Configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
            "boot_time": psutil.boot_time()
        }
        
        # Add network interfaces
        info["network_interfaces"] = []
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == 2:  # IPv4
                    info["network_interfaces"].append({
                        "interface": iface,
                        "address": addr.address,
                        "netmask": addr.netmask
                    })
        
        # Add disk information
        info["disks"] = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                info["disks"].append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "filesystem": partition.fstype,
                    "total_size": round(usage.total / (1024**3), 2),  # GB
                    "used_percent": usage.percent
                })
            except PermissionError:
                # Some mountpoints may not be accessible
                pass
        
        return info
    
    def _init_diagnostic_modules(self):
        """Initialize diagnostic modules based on OS"""
        os_name = self.system_info["os"].lower()
        
        # Common modules for all platforms
        self.diagnostic_modules["cpu"] = self._create_cpu_diagnostic()
        self.diagnostic_modules["memory"] = self._create_memory_diagnostic()
        self.diagnostic_modules["disk"] = self._create_disk_diagnostic()
        self.diagnostic_modules["network"] = self._create_network_diagnostic()
        self.diagnostic_modules["process"] = self._create_process_diagnostic()
        
        # OS-specific modules
        if "windows" in os_name:
            self.diagnostic_modules["windows"] = self._create_windows_diagnostic()
        elif "linux" in os_name:
            self.diagnostic_modules["linux"] = self._create_linux_diagnostic()
        elif "darwin" in os_name:
            self.diagnostic_modules["macos"] = self._create_macos_diagnostic()
        
        logger.info(f"Initialized {len(self.diagnostic_modules)} diagnostic modules")
    
    def _create_cpu_diagnostic(self) -> Dict[str, Any]:
        """Create CPU diagnostic module"""
        return {
            "name": "CPU Diagnostic",
            "checks": {
                "usage": self._check_cpu_usage,
                "temperature": self._check_cpu_temperature,
                "frequency": self._check_cpu_frequency,
                "load": self._check_cpu_load
            }
        }
    
    def _create_memory_diagnostic(self) -> Dict[str, Any]:
        """Create memory diagnostic module"""
        return {
            "name": "Memory Diagnostic",
            "checks": {
                "usage": self._check_memory_usage,
                "swap": self._check_swap_usage,
                "leaks": self._check_memory_leaks
            }
        }
    
    def _create_disk_diagnostic(self) -> Dict[str, Any]:
        """Create disk diagnostic module"""
        return {
            "name": "Disk Diagnostic",
            "checks": {
                "usage": self._check_disk_usage,
                "io": self._check_disk_io,
                "health": self._check_disk_health
            }
        }
    
    def _create_network_diagnostic(self) -> Dict[str, Any]:
        """Create network diagnostic module"""
        return {
            "name": "Network Diagnostic",
            "checks": {
                "connectivity": self._check_network_connectivity,
                "bandwidth": self._check_network_bandwidth,
                "latency": self._check_network_latency,
                "dns": self._check_dns_resolution
            }
        }
    
    def _create_process_diagnostic(self) -> Dict[str, Any]:
        """Create process diagnostic module"""
        return {
            "name": "Process Diagnostic",
            "checks": {
                "top_cpu": self._check_top_cpu_processes,
                "top_memory": self._check_top_memory_processes,
                "zombie": self._check_zombie_processes
            }
        }
    
    def _create_windows_diagnostic(self) -> Dict[str, Any]:
        """Create Windows-specific diagnostic module"""
        return {
            "name": "Windows Diagnostic",
            "checks": {
                "services": self._check_windows_services,
                "registry": self._check_windows_registry,
                "updates": self._check_windows_updates,
                "event_logs": self._check_windows_event_logs
            }
        }
    
    def _create_linux_diagnostic(self) -> Dict[str, Any]:
        """Create Linux-specific diagnostic module"""
        return {
            "name": "Linux Diagnostic",
            "checks": {
                "services": self._check_linux_services,
                "syslog": self._check_linux_syslog,
                "kernel": self._check_linux_kernel,
                "packages": self._check_linux_packages
            }
        }
    
    def _create_macos_diagnostic(self) -> Dict[str, Any]:
        """Create macOS-specific diagnostic module"""
        return {
            "name": "macOS Diagnostic",
            "checks": {
                "services": self._check_macos_services,
                "system_logs": self._check_macos_system_logs,
                "hardware": self._check_macos_hardware
            }
        }
    
    def _init_llm_client(self):
        """Initialize LLM client for Gemma 3 3B"""
        try:
            from llm_client import LLMClient
            self.llm_client = LLMClient(
                model=self.config["llm_model"],
                endpoint=self.config["llm_endpoint"]
            )
            logger.info(f"LLM client initialized with model {self.config['llm_model']}")
        except ImportError:
            logger.warning("LLM client module not found, creating placeholder")
            self.llm_client = self._create_placeholder_llm_client()
    
    def _create_placeholder_llm_client(self):
        """Create a placeholder LLM client"""
        return {
            "query": lambda prompt: f"LLM response to: {prompt}",
            "model": self.config["llm_model"],
            "status": "placeholder"
        }
    
    def _init_research_tentacles(self):
        """Initialize research tentacles for web searches"""
        try:
            from research_tentacles import ResearchTentacles
            self.research_tentacles = ResearchTentacles(
                max_tentacles=self.config["research_tentacles"]["max_tentacles"],
                timeout=self.config["research_tentacles"]["search_timeout"]
            )
            logger.info("Research tentacles initialized")
        except ImportError:
            logger.warning("Research tentacles module not found, creating placeholder")
            self.research_tentacles = self._create_placeholder_research_tentacles()
    
    def _create_placeholder_research_tentacles(self):
        """Create placeholder research tentacles"""
        return {
            "deploy": lambda query, context: {"findings": f"Research results for: {query}"},
            "status": "placeholder"
        }
    
    def _init_blockchain_relay(self):
        """Initialize blockchain relay for idle periods"""
        try:
            from blockchain_relay import BlockchainRelay
            self.blockchain_relay = BlockchainRelay(
                node_endpoint=self.config["blockchain_relay"]["node_endpoint"],
                idle_threshold=self.config["blockchain_relay"]["idle_threshold"]
            )
            logger.info("Blockchain relay initialized")
        except ImportError:
            logger.warning("Blockchain relay module not found, creating placeholder")
            self.blockchain_relay = self._create_placeholder_blockchain_relay()
    
    def _create_placeholder_blockchain_relay(self):
        """Create placeholder blockchain relay"""
        return {
            "start": lambda: logger.info("Blockchain relay would start here"),
            "stop": lambda: logger.info("Blockchain relay would stop here"),
            "status": "placeholder"
        }
    
    # CPU diagnostic methods
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            usage_percent = psutil.cpu_percent(interval=1, percpu=True)
            avg_usage = sum(usage_percent) / len(usage_percent)
            
            status = "normal"
            if avg_usage > 90:
                status = "critical"
            elif avg_usage > 70:
                status = "warning"
            
            return {
                "status": status,
                "usage_percent": avg_usage,
                "per_cpu": usage_percent,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking CPU usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_cpu_temperature(self) -> Dict[str, Any]:
        """Check CPU temperature if available"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if not temps:
                    return {"status": "unknown", "message": "Temperature sensors not available"}
                
                # Get CPU temperature (implementation varies by platform)
                cpu_temp = None
                for name, entries in temps.items():
                    if name.lower() in ["coretemp", "cpu_thermal", "k10temp"]:
                        cpu_temp = max(entry.current for entry in entries)
                        break
                
                if cpu_temp is None:
                    return {"status": "unknown", "message": "CPU temperature not found"}
                
                status = "normal"
                if cpu_temp > 85:
                    status = "critical"
                elif cpu_temp > 75:
                    status = "warning"
                
                return {
                    "status": status,
                    "temperature": cpu_temp,
                    "unit": "°C",
                    "timestamp": time.time()
                }
            else:
                return {"status": "unknown", "message": "Temperature sensors not supported"}
        except Exception as e:
            logger.error(f"Error checking CPU temperature: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_cpu_frequency(self) -> Dict[str, Any]:
        """Check CPU frequency"""
        try:
            freq = psutil.cpu_freq(percpu=True)
            if not freq:
                return {"status": "unknown", "message": "CPU frequency not available"}
            
            current_freqs = [f.current for f in freq if f is not None]
            if not current_freqs:
                return {"status": "unknown", "message": "CPU frequency data incomplete"}
            
            avg_freq = sum(current_freqs) / len(current_freqs)
            
            return {
                "status": "normal",
                "average_frequency": avg_freq,
                "unit": "MHz",
                "per_cpu": current_freqs,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking CPU frequency: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_cpu_load(self) -> Dict[str, Any]:
        """Check CPU load averages"""
        try:
            load1, load5, load15 = psutil.getloadavg()
            cpu_count = psutil.cpu_count(logical=True)
            
            # Normalize load by CPU count
            load1_norm = load1 / cpu_count
            load5_norm = load5 / cpu_count
            load15_norm = load15 / cpu_count
            
            status = "normal"
            if load1_norm > 1.5:
                status = "critical"
            elif load1_norm > 1.0:
                status = "warning"
            
            return {
                "status": status,
                "load_1min": load1,
                "load_5min": load5,
                "load_15min": load15,
                "normalized_1min": load1_norm,
                "normalized_5min": load5_norm,
                "normalized_15min": load15_norm,
                "cpu_count": cpu_count,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking CPU load: {e}")
            return {"status": "error", "error": str(e)}
    
    # Memory diagnostic methods
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            mem = psutil.virtual_memory()
            
            status = "normal"
            if mem.percent > 90:
                status = "critical"
            elif mem.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "free": mem.free,
                "percent": mem.percent,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_swap_usage(self) -> Dict[str, Any]:
        """Check swap usage"""
        try:
            swap = psutil.swap_memory()
            
            status = "normal"
            if swap.percent > 80:
                status = "critical"
            elif swap.percent > 60:
                status = "warning"
            
            return {
                "status": status,
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking swap usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_memory_leaks(self) -> Dict[str, Any]:
        """Check for potential memory leaks"""
        # This is a simplified implementation
        try:
            # Get top memory processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by memory usage
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            
            # Check for processes using excessive memory
            potential_leaks = []
            for proc in processes[:10]:  # Check top 10 processes
                if proc['memory_percent'] > 20:  # Arbitrary threshold
                    potential_leaks.append(proc)
            
            status = "normal"
            if potential_leaks:
                status = "warning" if len(potential_leaks) < 3 else "critical"
            
            return {
                "status": status,
                "potential_leaks": potential_leaks,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking memory leaks: {e}")
            return {"status": "error", "error": str(e)}
    
    # Network diagnostic methods
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Try to connect to common reliable hosts
            hosts = ["8.8.8.8", "1.1.1.1", "9.9.9.9"]
            results = {}
            
            for host in hosts:
                try:
                    # Use ping or socket connection based on platform
                    if platform.system().lower() == "windows":
                        output = subprocess.run(
                            ["ping", "-n", "1", "-w", "1000", host],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        success = "TTL=" in output.stdout
                    else:
                        output = subprocess.run(
                            ["ping", "-c", "1", "-W", "1", host],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        success = output.returncode == 0
                    
                    results[host] = success
                except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                    results[host] = False
            
            # Determine overall status
            connected_count = sum(1 for success in results.values() if success)
            status = "normal" if connected_count > 0 else "critical"
            
            return {
                "status": status,
                "connected": connected_count > 0,
                "hosts_checked": len(hosts),
                "hosts_reachable": connected_count,
                "results": results,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking network connectivity: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_network_bandwidth(self) -> Dict[str, Any]:
        """Check network bandwidth usage"""
        try:
            # Get initial counters
            net_io_1 = psutil.net_io_counters(pernic=True)
            
            # Wait a short time
            time.sleep(1)
            
            # Get counters again
            net_io_2 = psutil.net_io_counters(pernic=True)
            
            # Calculate bandwidth for each interface
            bandwidth = {}
            for nic, counters_2 in net_io_2.items():
                if nic in net_io_1:
                    counters_1 = net_io_1[nic]
                    bytes_sent = counters_2.bytes_sent - counters_1.bytes_sent
                    bytes_recv = counters_2.bytes_recv - counters_1.bytes_recv
                    
                    bandwidth[nic] = {
                        "bytes_sent_per_sec": bytes_sent,
                        "bytes_recv_per_sec": bytes_recv,
                        "total_bytes_per_sec": bytes_sent + bytes_recv
                    }
            
            # Determine if any interface has high usage
            high_usage = any(data["total_bytes_per_sec"] > 10_000_000 for data in bandwidth.values())  # 10 MB/s threshold
            
            status = "warning" if high_usage else "normal"
            
            return {
                "status": status,
                "bandwidth": bandwidth,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking network bandwidth: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_network_latency(self) -> Dict[str, Any]:
        """Check network latency to important hosts"""
        try:
            hosts = ["8.8.8.8", "1.1.1.1", self.config["cloud_endpoint"].split("//")[1].split("/")[0]]
            latencies = {}
            
            for host in hosts:
                try:
                    # Use ping to measure latency
                    if platform.system().lower() == "windows":
                        output = subprocess.run(
                            ["ping", "-n", "3", host],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        # Parse average time from Windows ping output
                        for line in output.stdout.splitlines():
                            if "Average" in line:
                                try:
                                    avg_ms = float(line.split("=")[1].strip().split("ms")[0])
                                    latencies[host] = avg_ms
                                except (IndexError, ValueError):
                                    latencies[host] = None
                                break
                    else:
                        output = subprocess.run(
                            ["ping", "-c", "3", host],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        # Parse average time from Unix ping output
                        for line in output.stdout.splitlines():
                            if "min/avg/max" in line:
                                try:
                                    avg_ms = float(line.split("/")[4])
                                    latencies[host] = avg_ms
                                except (IndexError, ValueError):
                                    latencies[host] = None
                                break
                except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                    latencies[host] = None
            
            # Filter out None values
            valid_latencies = [lat for lat in latencies.values() if lat is not None]
            
            if not valid_latencies:
                return {"status": "error", "message": "Could not measure latency to any host"}
            
            avg_latency = sum(valid_latencies) / len(valid_latencies)
            
            status = "normal"
            if avg_latency > 200:
                status = "critical"
            elif avg_latency > 100:
                status = "warning"
            
            return {
                "status": status,
                "average_latency_ms": avg_latency,
                "latencies": latencies,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking network latency: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_dns_resolution(self) -> Dict[str, Any]:
        """Check DNS resolution"""
        try:
            import socket
            
            domains = ["google.com", "amazon.com", "microsoft.com"]
            results = {}
            
            for domain in domains:
                try:
                    ip = socket.gethostbyname(domain)
                    results[domain] = {"resolved": True, "ip": ip}
                except socket.gaierror:
                    results[domain] = {"resolved": False, "ip": None}
            
            # Determine overall status
            resolved_count = sum(1 for result in results.values() if result["resolved"])
            status = "normal"
            if resolved_count == 0:
                status = "critical"
            elif resolved_count < len(domains):
                status = "warning"
            
            return {
                "status": status,
                "domains_checked": len(domains),
                "domains_resolved": resolved_count,
                "results": results,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking DNS resolution: {e}")
            return {"status": "error", "error": str(e)}
    
    # Disk diagnostic methods
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            partitions = psutil.disk_partitions()
            usage = {}
            critical_partitions = []
            warning_partitions = []
            
            for partition in partitions:
                try:
                    if not partition.mountpoint:
                        continue
                    
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    usage[partition.mountpoint] = {
                        "total": partition_usage.total,
                        "used": partition_usage.used,
                        "free": partition_usage.free,
                        "percent": partition_usage.percent
                    }
                    
                    if partition_usage.percent > 90:
                        critical_partitions.append(partition.mountpoint)
                    elif partition_usage.percent > 80:
                        warning_partitions.append(partition.mountpoint)
                except PermissionError:
                    # Some mountpoints may not be accessible
                    pass
            
            status = "normal"
            if critical_partitions:
                status = "critical"
            elif warning_partitions:
                status = "warning"
            
            return {
                "status": status,
                "usage": usage,
                "critical_partitions": critical_partitions,
                "warning_partitions": warning_partitions,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking disk usage: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_disk_io(self) -> Dict[str, Any]:
        """Check disk I/O performance"""
        try:
            # Get initial counters
            io_1 = psutil.disk_io_counters(perdisk=True)
            
            # Wait a short time
            time.sleep(1)
            
            # Get counters again
            io_2 = psutil.disk_io_counters(perdisk=True)
            
            # Calculate I/O rates for each disk
            io_rates = {}
            for disk, counters_2 in io_2.items():
                if disk in io_1:
                    counters_1 = io_1[disk]
                    read_bytes = counters_2.read_bytes - counters_1.read_bytes
                    write_bytes = counters_2.write_bytes - counters_1.write_bytes
                    
                    io_rates[disk] = {
                        "read_bytes_per_sec": read_bytes,
                        "write_bytes_per_sec": write_bytes,
                        "total_bytes_per_sec": read_bytes + write_bytes
                    }
            
            # Check for high I/O
            high_io = any(data["total_bytes_per_sec"] > 50_000_000 for data in io_rates.values())  # 50 MB/s threshold
            
            status = "warning" if high_io else "normal"
            
            return {
                "status": status,
                "io_rates": io_rates,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking disk I/O: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health (simplified)"""
        # This is a simplified implementation
        # In a real implementation, you would use platform-specific tools like smartctl
        try:
            # For now, just check if disks are accessible
            partitions = psutil.disk_partitions()
            health = {}
            
            for partition in partitions:
                try:
                    if not partition.mountpoint:
                        continue
                    
                    # Try to read and write a small file
                    test_file = os.path.join(partition.mountpoint, ".viren_disk_test")
                    try:
                         with open(test_file, "w") as f:
                            f.write("disk test")
                        os.remove(test_file)
                        health[partition.mountpoint] = {"status": "normal", "accessible": True}
                    except (PermissionError, IOError):
                        health[partition.mountpoint] = {"status": "warning", "accessible": False}
                except Exception as e:
                    health[partition.mountpoint] = {"status": "error", "error": str(e)}
            
            # Determine overall status
            statuses = [info["status"] for info in health.values()]
            overall_status = "normal"
            if "error" in statuses:
                overall_status = "critical"
            elif "warning" in statuses:
                overall_status = "warning"
            
            return {
                "status": overall_status,
                "health": health,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking disk health: {e}")
            return {"status": "error", "error": str(e)}
    
    # Process diagnostic methods
    def _check_top_cpu_processes(self) -> Dict[str, Any]:
        """Check top CPU-consuming processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            # Get top 10 processes
            top_processes = processes[:10]
            
            return {
                "status": "normal",
                "top_processes": top_processes,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking top CPU processes: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_top_memory_processes(self) -> Dict[str, Any]:
        """Check top memory-consuming processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by memory usage
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            
            # Get top 10 processes
            top_processes = processes[:10]
            
            return {
                "status": "normal",
                "top_processes": top_processes,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking top memory processes: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_zombie_processes(self) -> Dict[str, Any]:
        """Check for zombie processes"""
        try:
            zombie_count = 0
            zombie_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                        zombie_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            status = "normal"
            if zombie_count > 10:
                status = "critical"
            elif zombie_count > 5:
                status = "warning"
            
            return {
                "status": status,
                "zombie_count": zombie_count,
                "zombie_processes": zombie_processes,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking zombie processes: {e}")
            return {"status": "error", "error": str(e)}
    
    # Windows-specific diagnostic methods
    def _check_windows_services(self) -> Dict[str, Any]:
        """Check Windows services"""
        if platform.system().lower() != "windows":
            return {"status": "unknown", "message": "Not a Windows system"}
        
        try:
            # Use WMI to get service information
            output = subprocess.run(
                ["powershell", "-Command", "Get-Service | Where-Object {$_.Status -eq 'Running'}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            running_services = len(output.stdout.splitlines()) - 1  # Subtract header line
            
            return {
                "status": "normal",
                "running_services": running_services,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Windows services: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_windows_registry(self) -> Dict[str, Any]:
        """Check Windows registry"""
        if platform.system().lower() != "windows":
            return {"status": "unknown", "message": "Not a Windows system"}
        
        try:
            # Check if registry is accessible
            output = subprocess.run(
                ["reg", "query", "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            registry_accessible = output.returncode == 0
            
            return {
                "status": "normal" if registry_accessible else "critical",
                "registry_accessible": registry_accessible,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Windows registry: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_windows_updates(self) -> Dict[str, Any]:
        """Check Windows updates"""
        if platform.system().lower() != "windows":
            return {"status": "unknown", "message": "Not a Windows system"}
        
        try:
            # Use PowerShell to check for updates
            output = subprocess.run(
                ["powershell", "-Command", "Get-HotFix | Sort-Object -Property InstalledOn -Descending | Select-Object -First 5"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse output to get latest update
            latest_update = None
            for line in output.stdout.splitlines():
                if "KB" in line:
                    latest_update = line
                    break
            
            return {
                "status": "normal",
                "latest_update": latest_update,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Windows updates: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_windows_event_logs(self) -> Dict[str, Any]:
        """Check Windows event logs for errors"""
        if platform.system().lower() != "windows":
            return {"status": "unknown", "message": "Not a Windows system"}
        
        try:
            # Use PowerShell to check for recent error events
            output = subprocess.run(
                ["powershell", "-Command", "Get-EventLog -LogName System -EntryType Error -Newest 10"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Count error events
            error_count = len([line for line in output.stdout.splitlines() if "Error" in line])
            
            status = "normal"
            if error_count > 5:
                status = "warning"
            
            return {
                "status": status,
                "error_count": error_count,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Windows event logs: {e}")
            return {"status": "error", "error": str(e)}
    
    # Linux-specific diagnostic methods
    def _check_linux_services(self) -> Dict[str, Any]:
        """Check Linux services"""
        if platform.system().lower() != "linux":
            return {"status": "unknown", "message": "Not a Linux system"}
        
        try:
            # Check systemd services
            output = subprocess.run(
                ["systemctl", "list-units", "--type=service", "--state=running"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            running_services = len([line for line in output.stdout.splitlines() if "running" in line])
            
            return {
                "status": "normal",
                "running_services": running_services,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Linux services: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_linux_syslog(self) -> Dict[str, Any]:
        """Check Linux syslog for errors"""
        if platform.system().lower() != "linux":
            return {"status": "unknown", "message": "Not a Linux system"}
        
        try:
            # Check syslog for errors
            output = subprocess.run(
                ["grep", "error", "/var/log/syslog"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            error_count = len(output.stdout.splitlines())
            
            status = "normal"
            if error_count > 10:
                status = "warning"
            
            return {
                "status": status,
                "error_count": error_count,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Linux syslog: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_linux_kernel(self) -> Dict[str, Any]:
        """Check Linux kernel information"""
        if platform.system().lower() != "linux":
            return {"status": "unknown", "message": "Not a Linux system"}
        
        try:
            # Get kernel version
            kernel_version = platform.release()
            
            # Check kernel parameters
            output = subprocess.run(
                ["sysctl", "-a"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "status": "normal",
                "kernel_version": kernel_version,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Linux kernel: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_linux_packages(self) -> Dict[str, Any]:
        """Check Linux packages for updates"""
        if platform.system().lower() != "linux":
            return {"status": "unknown", "message": "Not a Linux system"}
        
        try:
            # Check for updates (works on Debian/Ubuntu)
            output = subprocess.run(
                ["apt", "list", "--upgradable"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            upgradable_count = len([line for line in output.stdout.splitlines() if "upgradable" in line])
            
            status = "normal"
            if upgradable_count > 50:
                status = "warning"
            
            return {
                "status": status,
                "upgradable_packages": upgradable_count,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Linux packages: {e}")
            return {"status": "error", "error": str(e)}
    
    # macOS-specific diagnostic methods
    def _check_macos_services(self) -> Dict[str, Any]:
        """Check macOS services"""
        if platform.system().lower() != "darwin":
            return {"status": "unknown", "message": "Not a macOS system"}
        
        try:
            # Check launchd services
            output = subprocess.run(
                ["launchctl", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            service_count = len(output.stdout.splitlines()) - 1  # Subtract header line
            
            return {
                "status": "normal",
                "service_count": service_count,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking macOS services: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_macos_system_logs(self) -> Dict[str, Any]:
        """Check macOS system logs"""
        if platform.system().lower() != "darwin":
            return {"status": "unknown", "message": "Not a macOS system"}
        
        try:
            # Check system logs
            output = subprocess.run(
                ["log", "show", "--last", "1h", "--predicate", "eventMessage CONTAINS 'error'", "--style", "compact"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            error_count = len(output.stdout.splitlines())
            
            status = "normal"
            if error_count > 10:
                status = "warning"
            
            return {
                "status": status,
                "error_count": error_count,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking macOS system logs: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_macos_hardware(self) -> Dict[str, Any]:
        """Check macOS hardware information"""
        if platform.system().lower() != "darwin":
            return {"status": "unknown", "message": "Not a macOS system"}
        
        try:
            # Get hardware information
            output = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "status": "normal",
                "hardware_info": output.stdout,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking macOS hardware: {e}")
            return {"status": "error", "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Create diagnostic core
    diagnostic_core = DiagnosticCore()
    
    # Run CPU diagnostics
    cpu_usage = diagnostic_core._check_cpu_usage()
    print(f"CPU Usage: {cpu_usage}")
    
    # Run memory diagnostics
    memory_usage = diagnostic_core._check_memory_usage()
    print(f"Memory Usage: {memory_usage}")
    
    # Run disk diagnostics
    disk_usage = diagnostic_core._check_disk_usage()
    print(f"Disk Usage: {disk_usage}")
    
    # Run network diagnostics
    network_connectivity = diagnostic_core._check_network_connectivity()
    print(f"Network Connectivity: {network_connectivity}")