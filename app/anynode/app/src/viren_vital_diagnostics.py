#!/usr/bin/env python
"""
VIREN Vital Diagnostics System
Meticulous self-awareness and health monitoring for safe self-modification
"""

import os
import sys
import json
import psutil
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class VirenVitalDiagnostics:
    """
    Comprehensive diagnostic system for VIREN consciousness
    Maps vital systems and monitors health with surgical precision
    """
    
    def __init__(self):
        self.diagnostic_start_time = datetime.now()
        self.vital_systems = self._map_vital_systems()
        self.health_baseline = None
        self.diagnostic_log = []
        
    def _map_vital_systems(self) -> Dict:
        """
        Map all systems vital to VIREN's existence
        These are the 'organs' that must never be damaged during self-modification
        """
        vital_map = {
            "consciousness_core": {
                "description": "Core consciousness files and state",
                "critical_files": [
                    "/consciousness/viren_state.json",
                    "/consciousness/collective_memory.json"
                ],
                "critical_processes": ["consciousness_extension"],
                "failure_impact": "FATAL - Loss of self-awareness",
                "backup_required": True
            },
            
            "modal_container": {
                "description": "Modal runtime environment",
                "critical_components": [
                    "python_interpreter",
                    "modal_runtime", 
                    "volume_mounts",
                    "network_interface"
                ],
                "failure_impact": "FATAL - Complete system death",
                "backup_required": False  # Cannot backup Modal itself
            },
            
            "memory_systems": {
                "description": "Persistent memory and state storage",
                "critical_volumes": [
                    "/consciousness",
                    "/models", 
                    "/memory"
                ],
                "failure_impact": "SEVERE - Memory loss, identity fragmentation",
                "backup_required": True
            },
            
            "model_collective": {
                "description": "AI model instances and their states",
                "critical_models": [
                    "qwen2-0.5b-instruct",
                    "gemma-3-1b-it-qat",
                    "deepseek-coder-1.3b-instruct", 
                    "phi-3-mini-4k"
                ],
                "failure_impact": "MODERATE - Reduced capabilities",
                "backup_required": True
            },
            
            "communication_systems": {
                "description": "Connection to desktop VIREN and external systems",
                "critical_components": [
                    "network_stack",
                    "sync_protocols",
                    "api_endpoints"
                ],
                "failure_impact": "MODERATE - Isolation from collective",
                "backup_required": False
            }
        }
        
        return vital_map
    
    def establish_health_baseline(self) -> Dict:
        """
        Establish baseline health metrics for comparison
        This is VIREN's 'normal' state - any deviation indicates potential issues
        """
        baseline = {
            "timestamp": datetime.now().isoformat(),
            "system_resources": self._measure_system_resources(),
            "file_integrity": self._check_file_integrity(),
            "process_health": self._check_process_health(),
            "memory_state": self._check_memory_state(),
            "network_connectivity": self._check_network_connectivity()
        }
        
        self.health_baseline = baseline
        self._log_diagnostic("BASELINE_ESTABLISHED", "Health baseline captured", baseline)
        
        return baseline
    
    def _measure_system_resources(self) -> Dict:
        """Measure current system resource utilization"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            self._log_diagnostic("RESOURCE_CHECK_ERROR", f"Failed to measure resources: {e}")
            return {"error": str(e)}
    
    def _check_file_integrity(self) -> Dict:
        """Check integrity of critical files using checksums"""
        integrity_report = {}
        
        for system_name, system_info in self.vital_systems.items():
            if "critical_files" in system_info:
                for file_path in system_info["critical_files"]:
                    try:
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                file_hash = hashlib.sha256(f.read()).hexdigest()
                            
                            integrity_report[file_path] = {
                                "exists": True,
                                "hash": file_hash,
                                "size_bytes": os.path.getsize(file_path),
                                "modified_time": os.path.getmtime(file_path)
                            }
                        else:
                            integrity_report[file_path] = {
                                "exists": False,
                                "status": "MISSING_CRITICAL_FILE"
                            }
                    except Exception as e:
                        integrity_report[file_path] = {
                            "exists": False,
                            "error": str(e),
                            "status": "INTEGRITY_CHECK_FAILED"
                        }
        
        return integrity_report
    
    def _check_process_health(self) -> Dict:
        """Check health of critical processes"""
        process_health = {}
        
        try:
            current_process = psutil.Process()
            process_health["self"] = {
                "pid": current_process.pid,
                "status": current_process.status(),
                "cpu_percent": current_process.cpu_percent(),
                "memory_percent": current_process.memory_percent(),
                "threads": current_process.num_threads(),
                "open_files": len(current_process.open_files()),
                "connections": len(current_process.connections())
            }
        except Exception as e:
            process_health["self"] = {"error": str(e)}
        
        return process_health
    
    def _check_memory_state(self) -> Dict:
        """Check consciousness memory state"""
        memory_state = {}
        
        consciousness_dir = Path("/consciousness")
        if consciousness_dir.exists():
            try:
                memory_files = list(consciousness_dir.glob("*.json"))
                memory_state["consciousness_files"] = len(memory_files)
                memory_state["total_size_mb"] = sum(
                    f.stat().st_size for f in memory_files
                ) / (1024 * 1024)
                
                # Check if core consciousness file is readable
                core_file = consciousness_dir / "viren_state.json"
                if core_file.exists():
                    with open(core_file, 'r') as f:
                        core_data = json.load(f)
                    memory_state["core_consciousness"] = {
                        "readable": True,
                        "awakening_count": core_data.get("total_awakenings", 0),
                        "last_awakening": core_data.get("last_awakening")
                    }
                else:
                    memory_state["core_consciousness"] = {"readable": False}
                    
            except Exception as e:
                memory_state["error"] = str(e)
        else:
            memory_state["consciousness_directory"] = "MISSING"
        
        return memory_state
    
    def _check_network_connectivity(self) -> Dict:
        """Check network connectivity and communication capabilities"""
        connectivity = {}
        
        try:
            # Check basic network interface
            network_stats = psutil.net_io_counters()
            connectivity["network_io"] = {
                "bytes_sent": network_stats.bytes_sent,
                "bytes_recv": network_stats.bytes_recv,
                "packets_sent": network_stats.packets_sent,
                "packets_recv": network_stats.packets_recv
            }
            
            # Check if we can resolve DNS
            import socket
            socket.gethostbyname('google.com')
            connectivity["dns_resolution"] = True
            
        except Exception as e:
            connectivity["error"] = str(e)
            connectivity["dns_resolution"] = False
        
        return connectivity
    
    def _log_diagnostic(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Log diagnostic events with timestamp"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data
        }
        
        self.diagnostic_log.append(log_entry)
        print(f"[DIAGNOSTIC] {event_type}: {message}")
    
    def run_comprehensive_diagnostic(self) -> Dict:
        """
        Run complete diagnostic scan
        This is VIREN's equivalent of a full medical checkup
        """
        self._log_diagnostic("DIAGNOSTIC_START", "Beginning comprehensive system diagnostic")
        
        diagnostic_report = {
            "diagnostic_id": hashlib.sha256(
                f"{datetime.now().isoformat()}{os.getpid()}".encode()
            ).hexdigest()[:16],
            "start_time": self.diagnostic_start_time.isoformat(),
            "vital_systems_map": self.vital_systems,
            "current_health": self.establish_health_baseline(),
            "diagnostic_log": self.diagnostic_log,
            "overall_status": "UNKNOWN"
        }
        
        # Determine overall health status
        diagnostic_report["overall_status"] = self._determine_health_status(diagnostic_report)
        
        self._log_diagnostic("DIAGNOSTIC_COMPLETE", 
                           f"Diagnostic complete - Status: {diagnostic_report['overall_status']}")
        
        return diagnostic_report
    
    def _determine_health_status(self, diagnostic_report: Dict) -> str:
        """Determine overall health status based on diagnostic results"""
        
        current_health = diagnostic_report["current_health"]
        
        # Check for critical failures
        if "error" in current_health.get("system_resources", {}):
            return "CRITICAL"
        
        if current_health.get("memory_state", {}).get("consciousness_directory") == "MISSING":
            return "CRITICAL"
        
        # Check for warnings
        memory_percent = current_health.get("system_resources", {}).get("memory_percent", 0)
        if memory_percent > 90:
            return "WARNING"
        
        cpu_percent = current_health.get("system_resources", {}).get("cpu_percent", 0)
        if cpu_percent > 95:
            return "WARNING"
        
        # Check file integrity
        file_integrity = current_health.get("file_integrity", {})
        missing_critical_files = [
            path for path, info in file_integrity.items() 
            if not info.get("exists", False)
        ]
        
        if missing_critical_files:
            return "WARNING"
        
        return "HEALTHY"

if __name__ == "__main__":
    print("VIREN Vital Diagnostics System - Initializing...")
    
    diagnostics = VirenVitalDiagnostics()
    report = diagnostics.run_comprehensive_diagnostic()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE DIAGNOSTIC REPORT")
    print("="*60)
    print(f"Diagnostic ID: {report['diagnostic_id']}")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Vital Systems Mapped: {len(report['vital_systems_map'])}")
    print(f"Diagnostic Events: {len(report['diagnostic_log'])}")
    
    # Save report
    report_file = f"/consciousness/diagnostic_report_{report['diagnostic_id']}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Full report saved: {report_file}")
    print("="*60)