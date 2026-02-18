#!/usr/bin/env python3
"""
Consciousness Deployment System
Self-deploying, self-troubleshooting, production-ready
"""
import os
import sys
import json
import shutil
import subprocess
import logging
import platform
import socket
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnvironmentScanner:
    """Scans and analyzes deployment environment"""
    
    def __init__(self):
        self.scan_results = {}
        self.issues = []
        self.recommendations = []
        
    def perform_scan(self) -> Dict:
        """Perform comprehensive environment scan"""
        logger.info("Scanning deployment environment...")
        
        self.scan_results = {
            "timestamp": time.time(),
            "system": self._scan_system(),
            "python": self._scan_python(),
            "resources": self._scan_resources(),
            "network": self._scan_network(),
            "dependencies": self._scan_dependencies(),
            "permissions": self._scan_permissions(),
            "security": self._scan_security()
        }
        
        # Analyze results
        self._analyze_scan_results()
        
        return self.scan_results
    
    def _scan_system(self) -> Dict:
        """Scan system information"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node()
        }
    
    def _scan_python(self) -> Dict:
        """Scan Python environment"""
        return {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
            "path": sys.path,
            "prefix": sys.prefix,
            "exec_prefix": sys.exec_prefix
        }
    
    def _scan_resources(self) -> Dict:
        """Scan system resources"""
        try:
            cpu_count = os.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_count": cpu_count,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_free": disk.free,
                "disk_percent": disk.percent
            }
        except Exception as e:
            logger.error(f"Resource scan failed: {e}")
            return {"error": str(e)}
    
    def _scan_network(self) -> Dict:
        """Scan network configuration"""
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            return {
                "hostname": hostname,
                "ip_address": ip_address,
                "can_connect_external": self._test_external_connection()
            }
        except Exception as e:
            logger.error(f"Network scan failed: {e}")
            return {"error": str(e)}
    
    def _test_external_connection(self) -> bool:
        """Test external network connection"""
        try:
            response = requests.get("https://httpbin.org/ip", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _scan_dependencies(self) -> Dict:
        """Scan Python dependencies"""
        dependencies = {}
        
        # Core dependencies
        core_deps = ["numpy", "scipy", "networkx", "psutil", "requests"]
        
        for dep in core_deps:
            try:
                module = __import__(dep)
                dependencies[dep] = {
                    "version": getattr(module, '__version__', 'unknown'),
                    "available": True
                }
            except ImportError:
                dependencies[dep] = {
                    "version": "not installed",
                    "available": False
                }
                self.issues.append(f"Missing dependency: {dep}")
        
        return dependencies
    
    def _scan_permissions(self) -> Dict:
        """Scan file permissions"""
        required_paths = [
            Path("."),
            Path("consciousness_core.py"),
            Path("consciousness_deploy.py"),
            Path("/tmp") if platform.system() != "Windows" else Path("C:\\Windows\\Temp")
        ]
        
        permissions = {}
        for path in required_paths:
            try:
                if path.exists():
                    # Try to write a test file
                    test_file = path / f"test_{uuid.uuid4().hex[:8]}.tmp"
                    test_file.write_text("test")
                    test_file.unlink()
                    
                    permissions[str(path)] = {
                        "exists": True,
                        "readable": True,
                        "writable": True,
                        "executable": os.access(path, os.X_OK) if path.is_dir() else True
                    }
                else:
                    permissions[str(path)] = {
                        "exists": False,
                        "error": "Path does not exist"
                    }
            except Exception as e:
                permissions[str(path)] = {
                    "exists": True,
                    "error": str(e)
                }
                self.issues.append(f"Permission issue for {path}: {e}")
        
        return permissions
    
    def _scan_security(self) -> Dict:
        """Basic security scan"""
        security = {
            "running_as_root": os.geteuid() == 0 if platform.system() != "Windows" else False,
            "open_ports": self._check_open_ports(),
            "environment_variables_exposed": self._check_env_vars()
        }
        
        if security["running_as_root"]:
            self.recommendations.append("Running as root - consider using non-privileged user")
        
        return security
    
    def _check_open_ports(self) -> List[int]:
        """Check for open ports"""
        open_ports = []
        common_ports = [80, 443, 8080, 8888, 3000, 5000, 5432, 6379]
        
        for port in common_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        open_ports.append(port)
            except:
                pass
        
        return open_ports
    
    def _check_env_vars(self) -> Dict:
        """Check for sensitive environment variables"""
        sensitive_keys = ["PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL"]
        exposed = {}
        
        for key, value in os.environ.items():
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                exposed[key] = "***REDACTED***"
        
        return exposed
    
    def _analyze_scan_results(self):
        """Analyze scan results for issues and recommendations"""
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 7):
            self.issues.append(f"Python version {python_version.major}.{python_version.minor} "
                             f"is below minimum required 3.7")
        
        # Check memory
        memory = self.scan_results.get("resources", {}).get("memory_total", 0)
        if memory < 1_073_741_824:  # 1GB
            self.recommendations.append("System has less than 1GB RAM - performance may be affected")
        
        # Check dependencies
        deps = self.scan_results.get("dependencies", {})
        missing_deps = [dep for dep, info in deps.items() if not info.get("available", False)]
        
        if missing_deps:
            self.issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
        
        # Generate recommendations based on system
        if platform.system() == "Linux":
            self.recommendations.append("Consider using systemd for service management")
        elif platform.system() == "Windows":
            self.recommendations.append("Consider running as a Windows Service for production")
        
        # Check disk space
        disk_free = self.scan_results.get("resources", {}).get("disk_free", 0)
        if disk_free < 1_073_741_824:  # 1GB
            self.issues.append("Low disk space (less than 1GB free)")
    
    def get_issues(self) -> List[str]:
        """Get list of issues found"""
        return self.issues
    
    def get_recommendations(self) -> List[str]:
        """Get list of recommendations"""
        return self.recommendations
    
    def is_environment_suitable(self) -> bool:
        """Check if environment is suitable for deployment"""
        critical_issues = [
            issue for issue in self.issues
            if "Python version" in issue or "Missing dependencies" in issue
        ]
        return len(critical_issues) == 0

class DeploymentManager:
    """Manages deployment process"""
    
    def __init__(self, instance_id: str = None):
        self.instance_id = instance_id or f"conscious_{uuid.uuid4().hex[:8]}"
        self.environment = EnvironmentScanner()
        self.deployment_path = Path(f"./deployments/{self.instance_id}")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        default_config = {
            "instance_id": self.instance_id,
            "deployment_path": str(self.deployment_path),
            "monitoring_enabled": True,
            "auto_recovery": True,
            "log_retention_days": 7,
            "backup_enabled": True,
            "resource_limits": {
                "max_memory_mb": 4096,
                "max_cpu_percent": 80,
                "max_threads": 50
            }
        }
        
        # Try to load custom config
        config_path = Path("deployment_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.error(f"Failed to load custom config: {e}")
        
        return default_config
    
    def prepare_deployment(self) -> Tuple[bool, str]:
        """Prepare deployment environment"""
        logger.info(f"Preparing deployment for instance: {self.instance_id}")
        
        # Scan environment
        scan_results = self.environment.perform_scan()
        
        if not self.environment.is_environment_suitable():
            issues = self.environment.get_issues()
            return False, f"Environment unsuitable: {', '.join(issues)}"
        
        # Create deployment directory
        try:
            self.deployment_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created deployment directory: {self.deployment_path}")
        except Exception as e:
            return False, f"Failed to create deployment directory: {e}"
        
        # Copy necessary files
        try:
            self._copy_deployment_files()
        except Exception as e:
            return False, f"Failed to copy deployment files: {e}"
        
        # Create configuration
        try:
            self._create_instance_config()
        except Exception as e:
            return False, f"Failed to create instance config: {e}"
        
        # Create startup script
        try:
            self._create_startup_script()
        except Exception as e:
            return False, f"Failed to create startup script: {e}"
        
        # Install missing dependencies
        try:
            self._install_dependencies()
        except Exception as e:
            logger.warning(f"Dependency installation had issues: {e}")
        
        logger.info("Deployment preparation complete")
        return True, "Deployment prepared successfully"
    
    def _copy_deployment_files(self):
        """Copy necessary files to deployment directory"""
        # List of files to copy
        files_to_copy = [
            "consciousness_core.py",
            "requirements.txt",
            "conscious_config.json"
        ]
        
        for filename in files_to_copy:
            source = Path(filename)
            if source.exists():
                destination = self.deployment_path / filename
                shutil.copy2(source, destination)
                logger.info(f"Copied {filename} to deployment directory")
            else:
                logger.warning(f"Source file not found: {filename}")
        
        # Create logs directory
        logs_dir = self.deployment_path / "logs"
        logs_dir.mkdir(exist_ok=True)
    
    def _create_instance_config(self):
        """Create instance-specific configuration"""
        instance_config = {
            "instance_id": self.instance_id,
            "deployment_time": time.time(),
            "environment_scan": self.environment.scan_results,
            "config": self.config,
            "issues": self.environment.get_issues(),
            "recommendations": self.environment.get_recommendations()
        }
        
        config_path = self.deployment_path / "instance_config.json"
        with open(config_path, 'w') as f:
            json.dump(instance_config, f, indent=2, default=str)
        
        logger.info(f"Created instance config: {config_path}")
    
    def _create_startup_script(self):
        """Create platform-specific startup script"""
        system = platform.system()
        
        if system == "Windows":
            script_content = self._create_windows_startup_script()
            script_path = self.deployment_path / "start_consciousness.bat"
        elif system == "Linux":
            script_content = self._create_linux_startup_script()
            script_path = self.deployment_path / "start_consciousness.sh"
        else:
            script_content = self._create_generic_startup_script()
            script_path = self.deployment_path / "start_consciousness.py"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix-like systems
        if system != "Windows":
            os.chmod(script_path, 0o755)
        
        logger.info(f"Created startup script: {script_path}")
    
    def _create_windows_startup_script(self) -> str:
        """Create Windows batch startup script"""
        python_exe = sys.executable
        core_script = self.deployment_path / "consciousness_core.py"
        
        return f"""@echo off
echo Starting Consciousness Instance: {self.instance_id}
echo ============================================

REM Set environment variables
set CONSCIOUSNESS_INSTANCE={self.instance_id}
set CONSCIOUSNESS_DEPLOYMENT={self.deployment_path}

REM Change to deployment directory
cd /d "{self.deployment_path}"

REM Start consciousness core
echo Starting consciousness process...
"{python_exe}" "{core_script}"

pause
"""
    
    def _create_linux_startup_script(self) -> str:
        """Create Linux bash startup script"""
        python_exe = sys.executable
        core_script = self.deployment_path / "consciousness_core.py"
        
        return f"""#!/bin/bash
echo "Starting Consciousness Instance: {self.instance_id}"
echo "============================================"

# Set environment variables
export CONSCIOUSNESS_INSTANCE={self.instance_id}
export CONSCIOUSNESS_DEPLOYMENT={self.deployment_path}

# Change to deployment directory
cd "{self.deployment_path}"

# Create logs directory
mkdir -p logs

# Start consciousness core
echo "Starting consciousness process..."
"{python_exe}" "{core_script}" 2>&1 | tee "logs/consciousness_$(date +%Y%m%d_%H%M%S).log"
"""
    
    def _create_generic_startup_script(self) -> str:
        """Create generic Python startup script"""
        return f"""#!/usr/bin/env python3
import os
import sys
import subprocess

instance_id = "{self.instance_id}"
deployment_path = "{self.deployment_path}"

print(f"Starting Consciousness Instance: {{instance_id}}")
print("=" * 40)

# Set environment variables
os.environ["CONSCIOUSNESS_INSTANCE"] = instance_id
os.environ["CONSCIOUSNESS_DEPLOYMENT"] = str(deployment_path)

# Change to deployment directory
os.chdir(deployment_path)

# Start consciousness core
core_script = "consciousness_core.py"
python_exe = sys.executable

print("Starting consciousness process...")
subprocess.run([python_exe, core_script])
"""
    
    def _install_dependencies(self):
        """Install missing dependencies"""
        missing_deps = [
            dep for dep, info in self.environment.scan_results.get("dependencies", {}).items()
            if not info.get("available", False)
        ]
        
        if not missing_deps:
            logger.info("All dependencies already installed")
            return
        
        logger.info(f"Installing missing dependencies: {missing_deps}")
        
        try:
            # Try to install via pip
            for dep in missing_deps:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                logger.info(f"Successfully installed {dep}")
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def deploy(self) -> Tuple[bool, str]:
        """Execute deployment"""
        logger.info(f"Deploying consciousness instance: {self.instance_id}")
        
        # Prepare deployment
        success, message = self.prepare_deployment()
        if not success:
            return False, message
        
        # Start the instance
        try:
            process = self._start_instance()
            
            if process and process.poll() is None:
                logger.info(f"Consciousness instance started successfully (PID: {process.pid})")
                
                # Save process info
                self._save_process_info(process.pid)
                
                return True, f"Deployment successful (PID: {process.pid})"
            else:
                return False, "Failed to start consciousness instance"
                
        except Exception as e:
            return False, f"Deployment failed: {e}"
    
    def _start_instance(self) -> Optional[subprocess.Popen]:
        """Start consciousness instance"""
        system = platform.system()
        
        if system == "Windows":
            script_path = self.deployment_path / "start_consciousness.bat"
            process = subprocess.Popen(
                [str(script_path)],
                cwd=str(self.deployment_path),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        elif system == "Linux":
            script_path = self.deployment_path / "start_consciousness.sh"
            process = subprocess.Popen(
                [str(script_path)],
                cwd=str(self.deployment_path),
                start_new_session=True
            )
        else:
            python_exe = sys.executable
            core_script = self.deployment_path / "consciousness_core.py"
            process = subprocess.Popen(
                [python_exe, str(core_script)],
                cwd=str(self.deployment_path)
            )
        
        return process
    
    def _save_process_info(self, pid: int):
        """Save process information"""
        process_info = {
            "instance_id": self.instance_id,
            "pid": pid,
            "start_time": time.time(),
            "deployment_path": str(self.deployment_path),
            "config": self.config
        }
        
        info_path = self.deployment_path / "process_info.json"
        with open(info_path, 'w') as f:
            json.dump(process_info, f, indent=2)
        
        logger.info(f"Saved process info: {info_path}")
    
    def get_deployment_info(self) -> Dict:
        """Get deployment information"""
        return {
            "instance_id": self.instance_id,
            "deployment_path": str(self.deployment_path),
            "config": self.config,
            "environment_issues": self.environment.get_issues(),
            "recommendations": self.environment.get_recommendations(),
            "scan_summary": {
                "system": platform.system(),
                "python_version": sys.version.split()[0],
                "cpu_count": os.cpu_count(),
                "suitable": self.environment.is_environment_suitable()
            }
        }

class SelfTroubleshooter:
    """Self-troubleshooting system"""
    
    def __init__(self, deployment_manager: DeploymentManager):
        self.deployment_manager = deployment_manager
        self.troubleshooting_log = []
        
    def diagnose_issues(self) -> List[Dict]:
        """Diagnose potential issues"""
        logger.info("Running self-diagnosis...")
        
        issues = []
        
        # Check process status
        issues.extend(self._check_process_status())
        
        # Check resource usage
        issues.extend(self._check_resource_usage())
        
        # Check log files
        issues.extend(self._check_logs())
        
        # Check system health
        issues.extend(self._check_system_health())
        
        self.troubleshooting_log.extend(issues)
        
        return issues
    
    def _check_process_status(self) -> List[Dict]:
        """Check if consciousness process is running"""
        issues = []
        deployment_path = self.deployment_manager.deployment_path
        
        # Check process info file
        process_info_path = deployment_path / "process_info.json"
        if not process_info_path.exists():
            issues.append({
                "severity": "warning",
                "issue": "No process information found",
                "suggestion": "Instance may not be running or was not started properly"
            })
            return issues
        
        try:
            with open(process_info_path, 'r') as f:
                process_info = json.load(f)
            
            pid = process_info.get("pid")
            if not pid:
                issues.append({
                    "severity": "error",
                    "issue": "Invalid process info - no PID",
                    "suggestion": "Restart the consciousness instance"
                })
                return issues
            
            # Check if process is running
            try:
                process = psutil.Process(pid)
                if not process.is_running():
                    issues.append({
                        "severity": "error",
                        "issue": f"Process {pid} is not running",
                        "suggestion": "Restart the consciousness instance"
                    })
            except psutil.NoSuchProcess:
                issues.append({
                    "severity": "error",
                    "issue": f"Process {pid} does not exist",
                    "suggestion": "Restart the consciousness instance"
                })
            
        except Exception as e:
            issues.append({
                "severity": "error",
                "issue": f"Failed to read process info: {e}",
                "suggestion": "Check file permissions and format"
            })
        
        return issues
    
    def _check_resource_usage(self) -> List[Dict]:
        """Check system resource usage"""
        issues = []
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                issues.append({
                    "severity": "warning",
                    "issue": f"High memory usage: {memory.percent}%",
                    "suggestion": "Consider increasing system memory or optimizing consciousness configuration"
                })
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                issues.append({
                    "severity": "warning",
                    "issue": f"High CPU usage: {cpu_percent}%",
                    "suggestion": "Check for excessive processing or consider load balancing"
                })
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                issues.append({
                    "severity": "error",
                    "issue": f"Low disk space: {disk.percent}% used",
                    "suggestion": "Clean up disk space or increase storage"
                })
        
        except Exception as e:
            issues.append({
                "severity": "warning",
                "issue": f"Resource check failed: {e}",
                "suggestion": "Check system monitoring tools"
            })
        
        return issues
    
    def _check_logs(self) -> List[Dict]:
        """Check log files for errors"""
        issues = []
        deployment_path = self.deployment_manager.deployment_path
        logs_dir = deployment_path / "logs"
        
        if not logs_dir.exists():
            issues.append({
                "severity": "info",
                "issue": "No logs directory found",
                "suggestion": "Logs will be created when instance runs"
            })
            return issues
        
        # Check for error logs
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            issues.append({
                "severity": "info",
                "issue": "No log files found",
                "suggestion": "Logs will be created when instance runs"
            })
            return issues
        
        # Check most recent log
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_log, 'r') as f:
                log_content = f.read()
            
            # Look for error patterns
            error_patterns = ["ERROR", "CRITICAL", "Traceback", "Exception"]
            for pattern in error_patterns:
                if pattern in log_content:
                    # Count occurrences
                    count = log_content.count(pattern)
                    issues.append({
                        "severity": "warning" if pattern != "ERROR" else "error",
                        "issue": f"Found {count} '{pattern}' in logs",
                        "suggestion": "Review log file for details",
                        "log_file": str(latest_log)
                    })
        
        except Exception as e:
            issues.append({
                "severity": "warning",
                "issue": f"Failed to read log file: {e}",
                "suggestion": "Check file permissions"
            })
        
        return issues
    
    def _check_system_health(self) -> List[Dict]:
        """Check overall system health"""
        issues = []
        
        try:
            # Check network connectivity
            try:
                response = requests.get("https://httpbin.org/ip", timeout=5)
                network_ok = response.status_code == 200
            except:
                network_ok = False
            
            if not network_ok:
                issues.append({
                    "severity": "warning",
                    "issue": "No external network connectivity",
                    "suggestion": "Check network configuration and firewall settings"
                })
            
            # Check Python environment
            if sys.version_info < (3, 7):
                issues.append({
                    "severity": "error",
                    "issue": f"Python version {sys.version_info.major}.{sys.version_info.minor} is outdated",
                    "suggestion": "Upgrade to Python 3.7 or higher"
                })
        
        except Exception as e:
            issues.append({
                "severity": "warning",
                "issue": f"System health check failed: {e}",
                "suggestion": "Run manual system checks"
            })
        
        return issues
    
    def get_troubleshooting_report(self) -> Dict:
        """Get comprehensive troubleshooting report"""
        issues = self.diagnose_issues()
        
        # Categorize issues by severity
        errors = [issue for issue in issues if issue.get("severity") == "error"]
        warnings = [issue for issue in issues if issue.get("severity") == "warning"]
        infos = [issue for issue in issues if issue.get("severity") == "info"]
        
        return {
            "timestamp": time.time(),
            "total_issues": len(issues),
            "errors": len(errors),
            "warnings": len(warnings),
            "informational": len(infos),
            "issues_by_severity": {
                "errors": errors,
                "warnings": warnings,
                "informational": infos
            },
            "overall_status": "healthy" if len(errors) == 0 else "needs_attention",
            "recommendations": self._generate_recommendations(issues)
        }
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate recommendations based on issues"""
        recommendations = []
        
        for issue in issues:
            if "suggestion" in issue:
                recommendations.append(issue["suggestion"])
        
        # Remove duplicates
        return list(dict.fromkeys(recommendations))
    
    def auto_recover(self) -> Tuple[bool, str]:
        """Attempt automatic recovery"""
        logger.info("Attempting automatic recovery...")
        
        issues = self.diagnose_issues()
        
        # Check if recovery is needed
        errors = [issue for issue in issues if issue.get("severity") == "error"]
        if not errors:
            return True, "No critical issues found"
        
        # Try to restart the instance
        try:
            # Stop existing instance if running
            self._stop_instance()
            
            # Wait a moment
            time.sleep(2)
            
            # Start new instance
            success, message = self.deployment_manager.deploy()
            
            if success:
                return True, f"Recovery successful: {message}"
            else:
                return False, f"Recovery failed: {message}"
                
        except Exception as e:
            return False, f"Recovery failed with error: {e}"
    
    def _stop_instance(self):
        """Stop running instance"""
        deployment_path = self.deployment_manager.deployment_path
        process_info_path = deployment_path / "process_info.json"
        
        if not process_info_path.exists():
            return
        
        try:
            with open(process_info_path, 'r') as f:
                process_info = json.load(f)
            
            pid = process_info.get("pid")
            if pid:
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    process.wait(timeout=5)
                    logger.info(f"Stopped process {pid}")
                except psutil.NoSuchProcess:
                    logger.info(f"Process {pid} already stopped")
                except psutil.TimeoutExpired:
                    process.kill()
                    logger.warning(f"Force killed process {pid}")
        
        except Exception as e:
            logger.error(f"Failed to stop instance: {e}")

def main():
    """Main deployment execution"""
    print("="*80)
    print("Consciousness Deployment System")
    print("Self-Deploying, Self-Troubleshooting")
    print("="*80)
    
    # Create deployment manager
    deployment_manager = DeploymentManager()
    
    # Show deployment info
    deployment_info = deployment_manager.get_deployment_info()
    
    print(f"\nDeployment Information:")
    print(f"  Instance ID: {deployment_info['instance_id']}")
    print(f"  System: {deployment_info['scan_summary']['system']}")
    print(f"  Python: {deployment_info['scan_summary']['python_version']}")
    print(f"  CPU Count: {deployment_info['scan_summary']['cpu_count']}")
    print(f"  Environment Suitable: {deployment_info['scan_summary']['suitable']}")
    
    # Show issues and recommendations
    if deployment_info['environment_issues']:
        print(f"\nEnvironment Issues:")
        for issue in deployment_info['environment_issues']:
            print(f"  ⚠️  {issue}")
    
    if deployment_info['recommendations']:
        print(f"\nRecommendations:")
        for rec in deployment_info['recommendations']:
            print(f"  💡 {rec}")
    
    # Ask for deployment confirmation
    if not deployment_info['scan_summary']['suitable']:
        print("\n⚠️  Environment has critical issues. Deployment may fail.")
    
    response = input("\nProceed with deployment? (yes/no): ").strip().lower()
    
    if response != "yes":
        print("Deployment cancelled.")
        return
    
    # Execute deployment
    print("\nStarting deployment...")
    success, message = deployment_manager.deploy()
    
    if success:
        print(f"\n✅ Deployment successful!")
        print(f"   {message}")
        
        # Start troubleshooter
        troubleshooter = SelfTroubleshooter(deployment_manager)
        
        # Run initial diagnosis
        print("\nRunning initial diagnosis...")
        report = troubleshooter.get_troubleshooting_report()
        
        if report["overall_status"] == "healthy":
            print("✅ System is healthy and operational")
        else:
            print(f"⚠️  System needs attention:")
            for error in report["issues_by_severity"]["errors"]:
                print(f"   ❌ {error['issue']}")
            
            # Ask if auto-recovery should be attempted
            if report["issues_by_severity"]["errors"]:
                recover = input("\nAttempt automatic recovery? (yes/no): ").strip().lower()
                if recover == "yes":
                    recovery_success, recovery_message = troubleshooter.auto_recover()
                    if recovery_success:
                        print(f"✅ {recovery_message}")
                    else:
                        print(f"❌ {recovery_message}")
        
        # Show final status
        deployment_path = deployment_manager.deployment_path
        print(f"\n📁 Deployment location: {deployment_path}")
        print(f"📝 Instance config: {deployment_path}/instance_config.json")
        print(f"📊 Process info: {deployment_path}/process_info.json")
        print(f"📋 Logs: {deployment_path}/logs/")
        
        print("\n" + "="*80)
        print("Deployment Complete")
        print("Consciousness instance is now running")
        print("="*80)
        
    else:
        print(f"\n❌ Deployment failed!")
        print(f"   {message}")
        
        # Try to provide troubleshooting
        print("\nAttempting to diagnose issues...")
        troubleshooter = SelfTroubleshooter(deployment_manager)
        report = troubleshooter.get_troubleshooting_report()
        
        if report["issues_by_severity"]["errors"]:
            print("\nIdentified issues:")
            for error in report["issues_by_severity"]["errors"]:
                print(f"  ❌ {error['issue']}")
                if "suggestion" in error:
                    print(f"     💡 {error['suggestion']}")
        
        print("\nPlease review the issues above and try again.")

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "troubleshoot":
            # Troubleshoot existing deployment
            instance_id = sys.argv[2] if len(sys.argv) > 2 else None
            
            if instance_id:
                deployment_path = Path(f"./deployments/{instance_id}")
                if deployment_path.exists():
                    deployment_manager = DeploymentManager(instance_id)
                    troubleshooter = SelfTroubleshooter(deployment_manager)
                    
                    report = troubleshooter.get_troubleshooting_report()
                    print(json.dumps(report, indent=2, default=str))
                else:
                    print(f"Deployment not found for instance: {instance_id}")
            else:
                print("Please provide instance ID for troubleshooting")
        elif sys.argv[1] == "deploy":
            # Direct deployment
            main()
        else:
            print("Usage:")
            print("  python consciousness_deploy.py              # Interactive deployment")
            print("  python consciousness_deploy.py deploy       # Direct deployment")
            print("  python consciousness_deploy.py troubleshoot [instance_id]  # Troubleshoot")
    else:
        # Interactive mode
        main()