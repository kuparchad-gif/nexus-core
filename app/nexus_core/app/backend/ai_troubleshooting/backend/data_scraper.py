# data_scraper.py
import json
import subprocess
import psutil
import platform
import os
import time
from pathlib import Path
import requests

class TroubleshootingDataScraper:
    def __init__(self):
        self.dataset_path = Path("datasets/viren_training")
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
    def continuous_scraping(self):
        """Run in background, consuming 10-20% system resources when idle"""
        while True:
            if self._is_system_idle():
                self._scrape_troubleshooting_data()
            time.sleep(60)  # Check every minute
    
    def _is_system_idle(self):
        """Check if system has resources available for training"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_available = psutil.virtual_memory().available / psutil.virtual_memory().total
        return cpu_usage < 60 and memory_available > 0.3
    
    def _scrape_troubleshooting_data(self):
        """Scrape pristine troubleshooting data from multiple sources"""
        print("🔍 Scraping troubleshooting data...")
        
        # Multi-platform system data
        self._scrape_system_forensics()
        
        # Cloud deployment scenarios
        self._scrape_cloud_deployment_data()
        
        # Network and enterprise scenarios
        self._scrape_enterprise_data()
        
        # Real-world error patterns
        self._scrape_error_patterns()
    
    def _scrape_system_forensics(self):
        """Scrape detailed system state across platforms"""
        system_data = {
            "timestamp": time.time(),
            "platform": platform.system(),
            "architecture": platform.architecture(),
            "system_metrics": self._get_system_metrics(),
            "docker_forensics": self._get_docker_forensics(),
            "network_state": self._get_network_state(),
            "service_health": self._get_service_health()
        }
        
        self._save_to_dataset("system_forensics", system_data)
    
    def _scrape_cloud_deployment_data(self):
        """Scrape cloud deployment scenarios"""
        cloud_scenarios = [
            {
                "scenario": "modal_deployment",
                "description": "Deploy container to Modal",
                "commands": ["modal deploy", "modal run", "modal volume create"],
                "troubleshooting": ["Check modal status", "Verify credentials", "Check resource limits"]
            },
            {
                "scenario": "aws_ecs_deployment", 
                "description": "Deploy to AWS ECS",
                "commands": ["aws ecs create-service", "aws ecr create-repository"],
                "troubleshooting": ["Check IAM permissions", "Verify VPC configuration", "Check ECS agent"]
            },
            {
                "scenario": "gcp_cloud_run",
                "description": "Deploy to Google Cloud Run",
                "commands": ["gcloud run deploy", "gcloud builds submit"],
                "troubleshooting": ["Check project permissions", "Verify container registry", "Check service account"]
            }
        ]
        
        for scenario in cloud_scenarios:
            self._save_to_dataset("cloud_deployments", scenario)
    
    def _scrape_enterprise_data(self):
        """Scrape enterprise networking and infrastructure scenarios"""
        enterprise_scenarios = {
            "network_troubleshooting": {
                "dns_issues": ["nslookup", "dig", "check /etc/resolv.conf"],
                "firewall_rules": ["iptables -L", "ufw status", "firewall-cmd --list-all"],
                "vpn_connectivity": ["ping gateway", "check routing table", "verify certificates"]
            },
            "kubernetes_operations": {
                "pod_failures": ["kubectl describe pod", "kubectl logs", "check resource limits"],
                "service_discovery": ["kubectl get endpoints", "check coreDNS", "verify network policies"],
                "storage_issues": ["check PVC", "verify storage class", "examine volume attachments"]
            },
            "security_incidents": {
                "access_issues": ["check audit logs", "verify permissions", "review sudoers"],
                "malware_detection": ["scan processes", "check crontab", "examine network connections"],
                "compliance_checks": ["verify encryption", "check patch levels", "audit configurations"]
            }
        }
        
        self._save_to_dataset("enterprise_scenarios", enterprise_scenarios)
    
    def _scrape_error_patterns(self):
        """Scrape real error messages and their solutions"""
        error_patterns = []
        
        # Common Docker errors
        error_patterns.extend([
            {
                "error": "Cannot connect to the Docker daemon",
                "solution": "sudo systemctl start docker",
                "explanation": "Docker service is not running"
            },
            {
                "error": "permission denied while trying to connect",
                "solution": "sudo usermod -aG docker $USER",
                "explanation": "User lacks Docker permissions"
            }
        ])
        
        # Cloud deployment errors
        error_patterns.extend([
            {
                "error": "ECR repository does not exist",
                "solution": "aws ecr create-repository --repository-name NAME",
                "explanation": "ECR repository needs to be created first"
            }
        ])
        
        self._save_to_dataset("error_patterns", error_patterns)
    
    def _get_system_metrics(self):
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def _get_docker_forensics(self):
        docker_info = {}
        try:
            # Docker installation
            docker_version = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            docker_info['installed'] = docker_version.returncode == 0
            
            if docker_info['installed']:
                # Running containers
                docker_ps = subprocess.run(['docker', 'ps', '--format', 'json'], capture_output=True, text=True)
                docker_info['containers'] = [json.loads(line) for line in docker_ps.stdout.strip().split('\n') if line]
                
                # Docker system info
                docker_info_json = subprocess.run(['docker', 'info', '--format', '{{json .}}'], capture_output=True, text=True)
                docker_info['system'] = json.loads(docker_info_json.stdout) if docker_info_json.stdout else {}
        except:
            pass
        
        return docker_info
    
    def _get_network_state(self):
        return {
            "connections": len(psutil.net_connections()),
            "interfaces": list(psutil.net_if_addrs().keys()),
            "io_counters": dict(psutil.net_io_counters()._asdict())
        }
    
    def _get_service_health(self):
        services = {}
        common_services = ['docker', 'nginx', 'ssh', 'redis', 'postgresql']
        
        for service in common_services:
            try:
                status = subprocess.run(['systemctl', 'is-active', service], capture_output=True, text=True)
                services[service] = status.stdout.strip()
            except:
                services[service] = 'unknown'
        
        return services
    
    def _save_to_dataset(self, category, data):
        """Save data to categorized JSONL files"""
        file_path = self.dataset_path / f"{category}.jsonl"
        with file_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')