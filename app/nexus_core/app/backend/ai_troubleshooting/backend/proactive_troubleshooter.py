import subprocess
import psutil
import platform
import time
import threading
import json

class SystemMonitor:
    def __init__(self, viren_agent=None):
        self.viren_agent = viren_agent
        self.monitoring = False
        self.alert_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85, 
            'disk_percent': 90,
            'docker_down': True,
            'high_load': True
        }
        self.last_alert_time = {}
        
    def start_monitoring(self):
        """Start background monitoring that alerts Viren when issues are detected"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        print("🔍 System Monitor: Started background monitoring")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        print("🔍 System Monitor: Stopped monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop that alerts Viren"""
        while self.monitoring:
            try:
                issues = self._check_system_health()
                if issues:
                    self._alert_viren(issues)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"🔍 Monitor Error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_system_health(self):
        """Check system for issues that need Viren's attention"""
        issues = []
        
        # CPU check
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > self.alert_thresholds['cpu_percent']:
            issues.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f'CPU usage at {cpu_usage}%',
                'details': self._get_top_processes()
            })
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > self.alert_thresholds['memory_percent']:
            issues.append({
                'type': 'high_memory', 
                'severity': 'warning',
                'message': f'Memory usage at {memory.percent}%',
                'available_gb': round(memory.available / (1024**3), 1)
            })
        
        # Disk check
        disk = psutil.disk_usage('/')
        if disk.percent > self.alert_thresholds['disk_percent']:
            issues.append({
                'type': 'low_disk',
                'severity': 'critical', 
                'message': f'Disk usage at {disk.percent}%',
                'free_gb': round(disk.free / (1024**3), 1)
            })
        
        # Docker check
        docker_status = self._check_docker_status()
        if not docker_status['running'] and self.alert_thresholds['docker_down']:
            issues.append({
                'type': 'docker_down',
                'severity': 'critical',
                'message': 'Docker service is not running',
                'details': docker_status
            })
        
        return issues
    
    def _alert_viren(self, issues):
        """Alert Viren about system issues"""
        for issue in issues:
            # Prevent spam - don't alert about same issue within 5 minutes
            issue_key = f"{issue['type']}_{issue['severity']}"
            last_alert = self.last_alert_time.get(issue_key, 0)
            
            if time.time() - last_alert > 300:  # 5 minutes
                print(f"🚨 MONITOR ALERT: {issue['message']}")
                
                # If Viren agent is connected, notify him
                if self.viren_agent:
                    self._notify_viren_agent(issue)
                
                self.last_alert_time[issue_key] = time.time()
    
    def _notify_viren_agent(self, issue):
        """Notify Viren agent about the issue"""
        try:
            # This would be the method Viren provides for monitor alerts
            if hasattr(self.viren_agent, 'handle_monitor_alert'):
                self.viren_agent.handle_monitor_alert(issue)
            else:
                print(f"🔍 Monitor: Viren alert method not available - {issue['message']}")
        except Exception as e:
            print(f"🔍 Monitor: Failed to alert Viren - {e}")
    
    def _check_docker_status(self):
        """Check if Docker is running"""
        status = {'installed': False, 'running': False}
        
        try:
            # Check installation
            docker_version = subprocess.run(['docker', '--version'], 
                                          capture_output=True, text=True, timeout=5)
            status['installed'] = docker_version.returncode == 0
            
            # Check if daemon is running
            docker_ps = subprocess.run(['docker', 'ps'], 
                                     capture_output=True, text=True, timeout=5)
            status['running'] = docker_ps.returncode == 0
            
        except Exception as e:
            status['error'] = str(e)
            
        return status
    
    def _get_top_processes(self, count=5):
        """Get top CPU using processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 1.0:
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage and return top ones
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            return processes[:count]
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_system_snapshot(self):
        """Get current system state snapshot"""
        return {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'docker_status': self._check_docker_status(),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def update_alert_thresholds(self, new_thresholds):
        """Update monitoring thresholds"""
        self.alert_thresholds.update(new_thresholds)
        print(f"🔍 Monitor: Updated thresholds - {new_thresholds}")