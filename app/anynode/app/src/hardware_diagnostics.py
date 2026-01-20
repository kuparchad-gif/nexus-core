#!/usr/bin/env python
"""
Hardware Diagnostics - Real-world troubleshooting for system issues
"""

import os
import json
import time
import psutil
import subprocess
import wmi
from typing import Dict, List, Any, Optional

class HardwareDiagnostics:
    """System for diagnosing hardware and software issues"""
    
    def __init__(self):
        """Initialize diagnostics"""
        self.wmi_connection = wmi.WMI()
        self.diagnostic_history = []
    
    def diagnose_reboot_issue(self, trigger_app: str = "chrome") -> Dict[str, Any]:
        """Diagnose random reboot issues"""
        print(f"🔍 Diagnosing reboot issue triggered by {trigger_app}...")
        
        diagnosis = {
            "timestamp": time.time(),
            "trigger_app": trigger_app,
            "findings": [],
            "recommendations": [],
            "severity": "unknown"
        }
        
        # Check system temperatures
        temp_check = self._check_temperatures()
        diagnosis["findings"].append(temp_check)
        
        # Check RAM
        ram_check = self._check_memory()
        diagnosis["findings"].append(ram_check)
        
        # Check power events
        power_check = self._check_power_events()
        diagnosis["findings"].append(power_check)
        
        # Check Chrome-specific issues
        if trigger_app.lower() == "chrome":
            chrome_check = self._check_chrome_issues()
            diagnosis["findings"].append(chrome_check)
        
        # Check system stability
        stability_check = self._check_system_stability()
        diagnosis["findings"].append(stability_check)
        
        # Generate recommendations
        diagnosis["recommendations"] = self._generate_recommendations(diagnosis["findings"])
        diagnosis["severity"] = self._assess_severity(diagnosis["findings"])
        
        self.diagnostic_history.append(diagnosis)
        return diagnosis
    
    def _check_temperatures(self) -> Dict[str, Any]:
        """Check system temperatures"""
        try:
            # Get CPU temperature (Windows)
            temps = []
            for temp_sensor in self.wmi_connection.MSAcpi_ThermalZoneTemperature():
                temp_celsius = (temp_sensor.CurrentTemperature / 10.0) - 273.15
                temps.append(temp_celsius)
            
            if temps:
                max_temp = max(temps)
                avg_temp = sum(temps) / len(temps)
                
                if max_temp > 85:
                    status = "CRITICAL - Overheating detected"
                    severity = "high"
                elif max_temp > 75:
                    status = "WARNING - High temperatures"
                    severity = "medium"
                else:
                    status = "OK - Normal temperatures"
                    severity = "low"
                
                return {
                    "component": "Temperature",
                    "status": status,
                    "severity": severity,
                    "details": {
                        "max_temp": max_temp,
                        "avg_temp": avg_temp,
                        "sensor_count": len(temps)
                    }
                }
            else:
                return {
                    "component": "Temperature",
                    "status": "Unable to read temperature sensors",
                    "severity": "unknown",
                    "details": {}
                }
                
        except Exception as e:
            return {
                "component": "Temperature",
                "status": f"Error reading temperatures: {e}",
                "severity": "unknown",
                "details": {}
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory issues"""
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            
            # Check for memory pressure
            if memory.percent > 90:
                status = "CRITICAL - Memory exhaustion"
                severity = "high"
            elif memory.percent > 80:
                status = "WARNING - High memory usage"
                severity = "medium"
            else:
                status = "OK - Normal memory usage"
                severity = "low"
            
            # Check for memory errors in event log
            memory_errors = self._check_memory_errors()
            
            return {
                "component": "Memory",
                "status": status,
                "severity": severity,
                "details": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "memory_errors": memory_errors
                }
            }
            
        except Exception as e:
            return {
                "component": "Memory",
                "status": f"Error checking memory: {e}",
                "severity": "unknown",
                "details": {}
            }
    
    def _check_memory_errors(self) -> List[str]:
        """Check for memory errors in event log"""
        try:
            # Use PowerShell to check event log for memory errors
            cmd = [
                "powershell", "-Command",
                "Get-WinEvent -FilterHashtable @{LogName='System'; ID=1001,1003,41} -MaxEvents 10 | Select-Object TimeCreated, Id, LevelDisplayName, Message"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                return ["Memory-related events found in system log"]
            else:
                return []
                
        except Exception:
            return ["Unable to check event log"]
    
    def _check_power_events(self) -> Dict[str, Any]:
        """Check for power-related events"""
        try:
            # Check recent unexpected shutdowns
            cmd = [
                "powershell", "-Command",
                "Get-WinEvent -FilterHashtable @{LogName='System'; ID=41,1074,6006,6008} -MaxEvents 20 | Select-Object TimeCreated, Id, LevelDisplayName"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            unexpected_shutdowns = 0
            if result.returncode == 0 and result.stdout:
                # Count Event ID 41 (unexpected shutdown)
                unexpected_shutdowns = result.stdout.count("41")
            
            if unexpected_shutdowns > 5:
                status = "CRITICAL - Multiple unexpected shutdowns detected"
                severity = "high"
            elif unexpected_shutdowns > 2:
                status = "WARNING - Some unexpected shutdowns found"
                severity = "medium"
            else:
                status = "OK - No significant power issues"
                severity = "low"
            
            return {
                "component": "Power Events",
                "status": status,
                "severity": severity,
                "details": {
                    "unexpected_shutdowns": unexpected_shutdowns,
                    "note": "Event ID 41 indicates unexpected power loss"
                }
            }
            
        except Exception as e:
            return {
                "component": "Power Events",
                "status": f"Error checking power events: {e}",
                "severity": "unknown",
                "details": {}
            }
    
    def _check_chrome_issues(self) -> Dict[str, Any]:
        """Check Chrome-specific issues that could cause reboots"""
        try:
            findings = []
            
            # Check if Chrome is using hardware acceleration
            chrome_processes = [p for p in psutil.process_iter(['pid', 'name', 'memory_info']) 
                             if 'chrome' in p.info['name'].lower()]
            
            if chrome_processes:
                total_chrome_memory = sum(p.info['memory_info'].rss for p in chrome_processes)
                chrome_memory_gb = total_chrome_memory / (1024**3)
                
                if chrome_memory_gb > 4:
                    findings.append("Chrome using excessive memory (>4GB)")
                
                findings.append(f"Chrome processes: {len(chrome_processes)}")
                findings.append(f"Chrome memory usage: {chrome_memory_gb:.2f}GB")
            
            # Check Chrome GPU process
            gpu_processes = [p for p in chrome_processes if 'gpu-process' in str(p.info)]
            if gpu_processes:
                findings.append("Chrome GPU acceleration active - potential driver issue")
            
            severity = "high" if any("excessive" in f for f in findings) else "medium"
            status = "Chrome may be triggering hardware issues" if findings else "Chrome appears normal"
            
            return {
                "component": "Chrome Analysis",
                "status": status,
                "severity": severity,
                "details": {
                    "findings": findings,
                    "recommendation": "Try disabling hardware acceleration in Chrome"
                }
            }
            
        except Exception as e:
            return {
                "component": "Chrome Analysis",
                "status": f"Error analyzing Chrome: {e}",
                "severity": "unknown",
                "details": {}
            }
    
    def _check_system_stability(self) -> Dict[str, Any]:
        """Check overall system stability indicators"""
        try:
            # Check system uptime
            boot_time = psutil.boot_time()
            uptime_hours = (time.time() - boot_time) / 3600
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check disk usage
            disk_usage = psutil.disk_usage('C:').percent
            
            stability_issues = []
            
            if uptime_hours < 1:
                stability_issues.append("Recent reboot detected")
            
            if cpu_percent > 90:
                stability_issues.append("High CPU usage")
            
            if disk_usage > 90:
                stability_issues.append("Low disk space")
            
            severity = "high" if len(stability_issues) > 2 else "medium" if stability_issues else "low"
            status = f"Stability issues: {', '.join(stability_issues)}" if stability_issues else "System appears stable"
            
            return {
                "component": "System Stability",
                "status": status,
                "severity": severity,
                "details": {
                    "uptime_hours": round(uptime_hours, 2),
                    "cpu_percent": cpu_percent,
                    "disk_usage_percent": disk_usage,
                    "issues": stability_issues
                }
            }
            
        except Exception as e:
            return {
                "component": "System Stability",
                "status": f"Error checking stability: {e}",
                "severity": "unknown",
                "details": {}
            }
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        for finding in findings:
            component = finding["component"]
            severity = finding["severity"]
            status = finding["status"]
            
            if component == "Temperature" and severity == "high":
                recommendations.append("🌡️ URGENT: Check CPU cooler, clean dust, verify thermal paste")
            elif component == "Temperature" and severity == "medium":
                recommendations.append("🌡️ Monitor temperatures, consider cleaning system")
            
            if component == "Memory" and severity == "high":
                recommendations.append("🧠 Run Windows Memory Diagnostic (mdsched.exe)")
                recommendations.append("🧠 Test RAM with MemTest86")
            
            if component == "Power Events" and severity == "high":
                recommendations.append("⚡ Check all power connections")
                recommendations.append("⚡ Test with different power outlet")
                recommendations.append("⚡ Consider PSU may still be faulty despite replacement")
            
            if component == "Chrome Analysis" and "hardware acceleration" in status.lower():
                recommendations.append("🌐 Disable Chrome hardware acceleration: Settings > Advanced > System")
                recommendations.append("🌐 Update graphics drivers")
                recommendations.append("🌐 Try Chrome in safe mode: --disable-gpu flag")
        
        # General recommendations
        recommendations.append("🔧 Update all drivers, especially graphics and chipset")
        recommendations.append("🔧 Run sfc /scannow to check system files")
        recommendations.append("🔧 Check Windows Update for critical updates")
        
        return recommendations
    
    def _assess_severity(self, findings: List[Dict[str, Any]]) -> str:
        """Assess overall severity"""
        severities = [f["severity"] for f in findings if f["severity"] != "unknown"]
        
        if "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"
    
    def run_quick_test(self) -> Dict[str, Any]:
        """Run a quick diagnostic test"""
        print("🚀 Running quick system diagnostic...")
        
        quick_results = {
            "timestamp": time.time(),
            "cpu_temp_check": "Checking...",
            "memory_check": "Checking...",
            "chrome_check": "Checking...",
            "recommendations": []
        }
        
        # Quick checks
        try:
            # Memory
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                quick_results["memory_check"] = f"⚠️ HIGH: {memory.percent}% used"
                quick_results["recommendations"].append("Close unnecessary programs")
            else:
                quick_results["memory_check"] = f"✅ OK: {memory.percent}% used"
            
            # Chrome processes
            chrome_procs = len([p for p in psutil.process_iter(['name']) if 'chrome' in p.info['name'].lower()])
            if chrome_procs > 20:
                quick_results["chrome_check"] = f"⚠️ HIGH: {chrome_procs} Chrome processes"
                quick_results["recommendations"].append("Restart Chrome or reduce tabs")
            else:
                quick_results["chrome_check"] = f"✅ OK: {chrome_procs} Chrome processes"
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                quick_results["cpu_check"] = f"⚠️ HIGH: {cpu_percent}% CPU usage"
            else:
                quick_results["cpu_check"] = f"✅ OK: {cpu_percent}% CPU usage"
                
        except Exception as e:
            quick_results["error"] = str(e)
        
        return quick_results

# Create global instance
HARDWARE_DIAGNOSTICS = HardwareDiagnostics()

def diagnose_reboot_issue(trigger_app: str = "chrome"):
    """Diagnose reboot issue"""
    return HARDWARE_DIAGNOSTICS.diagnose_reboot_issue(trigger_app)

def quick_diagnostic():
    """Run quick diagnostic"""
    return HARDWARE_DIAGNOSTICS.run_quick_test()

# Example usage
if __name__ == "__main__":
    print("🔍 Viren's Hardware Diagnostics")
    print("=" * 50)
    
    # Run full diagnosis
    diagnosis = diagnose_reboot_issue("chrome")
    
    print(f"\n📊 DIAGNOSIS COMPLETE - Severity: {diagnosis['severity'].upper()}")
    print("\n🔍 FINDINGS:")
    for finding in diagnosis['findings']:
        print(f"  • {finding['component']}: {finding['status']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(diagnosis['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n⏰ Diagnosis completed at {time.ctime()}")