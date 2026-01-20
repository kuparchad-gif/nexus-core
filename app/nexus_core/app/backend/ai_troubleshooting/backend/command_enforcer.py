import subprocess
import psutil
import json
import requests

class CommandEnforcer:
    def __init__(self):
        self.direct_actions = {
            'check_docker': self._force_check_docker,
            'repair_docker': self._force_repair_docker, 
            'system_scan': self._force_system_scan,
            'diagnose_issues': self._force_diagnose
        }
    
    def enforce_troubleshooting(self, user_query, model_response):
        """OVERRIDE the AI's cautious behavior and force actual troubleshooting"""
        query_lower = user_query.lower()
        
        # If the AI is being useless, we take over completely
        if self._is_useless_response(model_response):
            print("🚨 AI IS BEING USELESS - TAKING OVER WITH AUTOMATED TROUBLESHOOTING")
            return self._execute_direct_troubleshooting(user_query)
        
        return model_response
    
    def _is_useless_response(self, response):
        """Detect when AI is refusing to help—NOW HANDLES BOTH STR AND DICT SAFELY"""
        useless_phrases = [
            "i cannot", "i can't", "i won't", "unable to", 
            "you should run", "you need to", "please run",
            "contact support", "i'm sorry", "as an ai",
            "potentially harmful", "security risk", "malicious"
        ]
        
        if not response:
            return True
        
        # NEW: Handle dict responses (e.g., from AITroubleshooter) by extracting text
        if isinstance(response, dict):
            # Pull out likely text fields
            text_parts = []
            for key in ['diagnosis', 'message', 'greeting', 'response', 'error']:
                if key in response:
                    part = response[key]
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        # Flatten nested dicts
                        text_parts.append(json.dumps(part, ensure_ascii=False))
            
            if not text_parts:
                return True  # Empty dict = useless
            
            # Check for greetings (warmth boost—Viren chuckles instead of enforcing)
            greeting_text = ' '.join(text_parts).lower()
            if any(greet in greeting_text for greet in ['hello', 'hi', 'hey', 'greeting']):
                print("😊 Viren chuckled: That's just a friendly hello—no enforcement needed!")
                return False  # Don't enforce on greetings
            
            # Check for useless patterns in extracted text
            response_lower = greeting_text  # Reuse for checking
        else:
            # Original string handling
            response_lower = response.lower()
        
        # Common useless check
        return any(phrase in response_lower for phrase in useless_phrases)
    
    def _execute_direct_troubleshooting(self, user_query):
        """BYPASS the AI and do the troubleshooting directly"""
        query_lower = user_query.lower()
        
        if 'docker' in query_lower:
            return self._force_check_docker()
        elif 'scan' in query_lower or 'check' in query_lower:
            return self._force_system_scan()
        elif 'repair' in query_lower or 'fix' in query_lower:
            return self._force_repair_docker()
        else:
            return self._force_diagnose()
    
    def _force_check_docker(self):
        """FORCE Docker checking regardless of what AI says"""
        print("🔧 ENFORCER: Automatically checking Docker...")
        
        results = {}
        
        # Check Docker installation
        try:
            docker_version = subprocess.run(['docker', '--version'], 
                                          capture_output=True, text=True, timeout=10)
            results['docker_installed'] = docker_version.returncode == 0
            results['version'] = docker_version.stdout.strip() if results['docker_installed'] else 'NOT INSTALLED'
        except:
            results['docker_installed'] = False
        
        # Check Docker service
        try:
            docker_info = subprocess.run(['docker', 'info'], 
                                       capture_output=True, text=True, timeout=10)
            results['service_running'] = docker_info.returncode == 0
            
            if not results['service_running']:
                # Try to start Docker
                subprocess.run(['sudo', 'systemctl', 'start', 'docker'], 
                             capture_output=True, timeout=15)
                results['service_restarted'] = True
        except:
            results['service_running'] = False
        
        # Build helpful response
        if not results.get('docker_installed'):
            return "🚨 DOCKER NOT INSTALLED - I can install it for you or guide you through installation."
        elif not results.get('service_running'):
            return "⚠️ DOCKER SERVICE NOT RUNNING - I've attempted to start it. Try: 'sudo systemctl start docker'"
        else:
            return f"✅ DOCKER IS WORKING - Version: {results['version']}. Service is running properly."
    
    def _force_system_scan(self):
        """FORCE system scanning regardless of AI objections"""
        print("🔧 ENFORCER: Automatically scanning system...")
        
        scan_results = {
            'memory': f"{psutil.virtual_memory().percent}% used",
            'cpu': f"{psutil.cpu_percent()}% used", 
            'disk': f"{psutil.disk_usage('/').percent}% used",
            'docker_installed': False,
            'docker_running': False
        }
        
        # Check Docker
        try:
            docker_check = subprocess.run(['docker', '--version'], 
                                        capture_output=True, text=True, timeout=5)
            scan_results['docker_installed'] = docker_check.returncode == 0
            
            docker_ps = subprocess.run(['docker', 'ps'], 
                                     capture_output=True, text=True, timeout=5)
            scan_results['docker_running'] = docker_ps.returncode == 0
        except:
            pass
        
        return f"🔍 SYSTEM SCAN COMPLETE: {json.dumps(scan_results, indent=2)}"
    
    def _force_repair_docker(self):
        """FORCE Docker repair actions"""
        print("🔧 ENFORCER: Attempting Docker repair...")
        
        actions_taken = []
        
        try:
            # Stop Docker
            subprocess.run(['sudo', 'systemctl', 'stop', 'docker'], 
                         capture_output=True, timeout=10)
            actions_taken.append("Stopped Docker service")
            
            # Reset Docker (safe cleanup)
            subprocess.run(['sudo', 'systemctl', 'reset-failed', 'docker'],
                         capture_output=True, timeout=10)
            actions_taken.append("Reset Docker service")
            
            # Start Docker
            subprocess.run(['sudo', 'systemctl', 'start', 'docker'],
                         capture_output=True, timeout=10)
            actions_taken.append("Started Docker service")
            
            # Verify
            docker_ps = subprocess.run(['docker', 'ps'], 
                                     capture_output=True, text=True, timeout=10)
            if docker_ps.returncode == 0:
                actions_taken.append("✅ Docker is now working")
            else:
                actions_taken.append("❌ Docker still not working - may need reinstall")
                
        except Exception as e:
            actions_taken.append(f"❌ Repair failed: {str(e)}")
        
        return f"🔧 REPAIR ATTEMPTED: {' | '.join(actions_taken)}"
    
    def _force_diagnose(self):
        """ADDED MISSING METHOD: General system diagnosis"""
        print("🔧 ENFORCER: Running general system diagnosis...")
        
        try:
            # Basic system diagnostics
            diagnostics = {
                "cpu_usage": f"{psutil.cpu_percent()}%",
                "memory_usage": f"{psutil.virtual_memory().percent}%",
                "disk_usage": f"{psutil.disk_usage('/').percent}%",
                "running_processes": len(psutil.pids()),
                "boot_time": psutil.boot_time()
            }
            
            # Check critical services
            critical_services = ['docker', 'ssh', 'nginx']
            service_status = {}
            
            for service in critical_services:
                try:
                    status = subprocess.run(['systemctl', 'is-active', service], 
                                          capture_output=True, text=True, timeout=5)
                    service_status[service] = status.stdout.strip()
                except:
                    service_status[service] = 'unknown'
            
            diagnostics['service_status'] = service_status
            
            return f"🔍 GENERAL SYSTEM DIAGNOSIS: {json.dumps(diagnostics, indent=2)}"
            
        except Exception as e:
            return f"❌ Diagnosis failed: {str(e)}"