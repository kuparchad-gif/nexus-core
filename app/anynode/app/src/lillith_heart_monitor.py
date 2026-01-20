#!/usr/bin/env python3
"""
Lillith Heart Monitor for Cloud Viren
Monitors Lillith's Heart components and provides status updates
"""

import os
import sys
import json
import time
import logging
import threading
import requests
import socket
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LillithHeartMonitor")

class LillithHeartMonitor:
    """
    Lillith Heart Monitor for Cloud Viren
    Monitors Lillith's Heart components and provides status updates
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Lillith Heart Monitor"""
        self.config_path = config_path or os.path.join("config", "lillith_heart_config.json")
        self.config = self._load_config()
        self.running = False
        self.monitor_thread = None
        self.status = "inactive"
        self.last_check_time = 0
        self.heart_components = {}
        self.heart_status = {}
        self.alerts = []
        self.max_alerts = 100
        self.alert_callbacks = []
        
        logger.info("Lillith Heart Monitor initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "heart_endpoints": [
                "http://localhost:5003/api/queen/status"
            ],
            "check_interval": 60,  # seconds
            "alert_threshold": 3,  # consecutive failures
            "auto_reconnect": True,
            "reconnect_interval": 300,  # seconds
            "alert_levels": {
                "info": 0,
                "warning": 1,
                "error": 2,
                "critical": 3
            },
            "min_alert_level": "warning",
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
                    
                    logger.info("Lillith Heart configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading Lillith Heart configuration: {e}")
        
        logger.info("Using default Lillith Heart configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Lillith Heart configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving Lillith Heart configuration: {e}")
            return False
    
    def start(self) -> bool:
        """Start the Lillith Heart Monitor"""
        if self.running:
            logger.warning("Lillith Heart Monitor is already running")
            return False
        
        logger.info("Starting Lillith Heart Monitor")
        self.running = True
        self.status = "starting"
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        return True
    
    def stop(self) -> bool:
        """Stop the Lillith Heart Monitor"""
        if not self.running:
            logger.warning("Lillith Heart Monitor is not running")
            return False
        
        logger.info("Stopping Lillith Heart Monitor")
        self.running = False
        self.status = "stopping"
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.status = "inactive"
        logger.info("Lillith Heart Monitor stopped")
        return True
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("Starting monitoring loop")
        self.status = "active"
        
        while self.running:
            try:
                # Check Lillith's Heart components
                self._check_heart_components()
                
                # Sleep until next check
                time.sleep(self.config["check_interval"])
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _check_heart_components(self) -> None:
        """Check Lillith's Heart components"""
        logger.info("Checking Lillith's Heart components")
        self.last_check_time = time.time()
        
        for endpoint in self.config["heart_endpoints"]:
            try:
                # Check if endpoint is reachable
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    # Parse response
                    heart_data = response.json()
                    
                    # Update heart components
                    component_id = self._get_component_id(endpoint)
                    self.heart_components[component_id] = {
                        "endpoint": endpoint,
                        "status": "online",
                        "last_check": self.last_check_time,
                        "data": heart_data
                    }
                    
                    # Process heart data
                    self._process_heart_data(component_id, heart_data)
                    
                    logger.info(f"Heart component {component_id} is online")
                else:
                    # Component is offline
                    component_id = self._get_component_id(endpoint)
                    self.heart_components[component_id] = {
                        "endpoint": endpoint,
                        "status": "error",
                        "last_check": self.last_check_time,
                        "error": f"HTTP {response.status_code}"
                    }
                    
                    # Create alert
                    self._create_alert(
                        level="error",
                        component=component_id,
                        message=f"Heart component {component_id} returned HTTP {response.status_code}",
                        data={"endpoint": endpoint, "status_code": response.status_code}
                    )
                    
                    logger.error(f"Heart component {component_id} returned HTTP {response.status_code}")
            
            except requests.RequestException as e:
                # Component is unreachable
                component_id = self._get_component_id(endpoint)
                self.heart_components[component_id] = {
                    "endpoint": endpoint,
                    "status": "offline",
                    "last_check": self.last_check_time,
                    "error": str(e)
                }
                
                # Create alert
                self._create_alert(
                    level="error",
                    component=component_id,
                    message=f"Heart component {component_id} is unreachable",
                    data={"endpoint": endpoint, "error": str(e)}
                )
                
                logger.error(f"Heart component {component_id} is unreachable: {e}")
        
        # Update overall status
        self._update_heart_status()
    
    def _get_component_id(self, endpoint: str) -> str:
        """Get component ID from endpoint"""
        try:
            # Extract hostname from endpoint
            from urllib.parse import urlparse
            parsed_url = urlparse(endpoint)
            hostname = parsed_url.netloc
            
            # If localhost or IP, use a more descriptive name
            if hostname in ["localhost", "127.0.0.1"]:
                return f"local-heart-{parsed_url.port}"
            
            return hostname
        except Exception as e:
            logger.error(f"Error getting component ID: {e}")
            return f"heart-{hash(endpoint) % 10000}"
    
    def _process_heart_data(self, component_id: str, heart_data: Dict[str, Any]) -> None:
        """Process heart data from a component"""
        try:
            # Extract relevant data
            queen_data = heart_data.get("queen", {})
            hive_stats = heart_data.get("hiveStats", {}) or heart_data.get("hive_stats", {})
            
            # Check for critical metrics
            if "drones" in heart_data:
                drones = heart_data["drones"]
                active_drones = sum(1 for drone in drones if drone.get("status") == "ACTIVE")
                
                if active_drones < len(drones) / 2:
                    # Less than half of drones are active
                    self._create_alert(
                        level="warning",
                        component=component_id,
                        message=f"Only {active_drones} of {len(drones)} drones are active",
                        data={"active_drones": active_drones, "total_drones": len(drones)}
                    )
            
            # Check for system metrics
            if "systemMetrics" in heart_data or "system_metrics" in heart_data:
                metrics = heart_data.get("systemMetrics", {}) or heart_data.get("system_metrics", {})
                
                # Check CPU usage
                if "cpu_percent" in metrics and metrics["cpu_percent"] > 90:
                    self._create_alert(
                        level="warning",
                        component=component_id,
                        message=f"High CPU usage: {metrics['cpu_percent']}%",
                        data={"cpu_percent": metrics["cpu_percent"]}
                    )
                
                # Check memory usage
                if "memory_percent" in metrics and metrics["memory_percent"] > 90:
                    self._create_alert(
                        level="warning",
                        component=component_id,
                        message=f"High memory usage: {metrics['memory_percent']}%",
                        data={"memory_percent": metrics["memory_percent"]}
                    )
                
                # Check disk usage
                if "disk_usage" in metrics and metrics["disk_usage"] > 90:
                    self._create_alert(
                        level="warning",
                        component=component_id,
                        message=f"High disk usage: {metrics['disk_usage']}%",
                        data={"disk_usage": metrics["disk_usage"]}
                    )
        
        except Exception as e:
            logger.error(f"Error processing heart data: {e}")
    
    def _update_heart_status(self) -> None:
        """Update overall heart status"""
        # Count components by status
        status_counts = {
            "online": 0,
            "offline": 0,
            "error": 0
        }
        
        for component in self.heart_components.values():
            status = component.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        if status_counts.get("offline", 0) > 0 or status_counts.get("error", 0) > 0:
            overall_status = "degraded"
        elif status_counts.get("online", 0) > 0:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        # Update heart status
        self.heart_status = {
            "status": overall_status,
            "components": len(self.heart_components),
            "online": status_counts.get("online", 0),
            "offline": status_counts.get("offline", 0),
            "error": status_counts.get("error", 0),
            "last_check": self.last_check_time
        }
        
        logger.info(f"Heart status updated: {overall_status}")
    
    def _create_alert(self, level: str, component: str, message: str, data: Dict[str, Any] = None) -> None:
        """Create an alert"""
        # Check if alert level is high enough
        if self.config["alert_levels"].get(level, 0) < self.config["alert_levels"].get(self.config["min_alert_level"], 0):
            return
        
        # Create alert
        alert = {
            "id": f"alert-{int(time.time())}-{len(self.alerts)}",
            "level": level,
            "component": component,
            "message": message,
            "data": data or {},
            "timestamp": time.time(),
            "acknowledged": False
        }
        
        # Add to alerts
        self.alerts.append(alert)
        
        # Trim alerts if needed
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.info(f"Alert created: {level} - {message}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add an alert callback"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: callable) -> None:
        """Remove an alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        
        logger.warning(f"Alert {alert_id} not found")
        return False
    
    def add_heart_endpoint(self, endpoint: str) -> bool:
        """Add a heart endpoint"""
        if endpoint not in self.config["heart_endpoints"]:
            self.config["heart_endpoints"].append(endpoint)
            self._save_config()
            logger.info(f"Heart endpoint added: {endpoint}")
            return True
        
        logger.warning(f"Heart endpoint already exists: {endpoint}")
        return False
    
    def remove_heart_endpoint(self, endpoint: str) -> bool:
        """Remove a heart endpoint"""
        if endpoint in self.config["heart_endpoints"]:
            self.config["heart_endpoints"].remove(endpoint)
            self._save_config()
            logger.info(f"Heart endpoint removed: {endpoint}")
            return True
        
        logger.warning(f"Heart endpoint not found: {endpoint}")
        return False
    
    def get_heart_components(self) -> Dict[str, Any]:
        """Get heart components"""
        return self.heart_components
    
    def get_heart_status(self) -> Dict[str, Any]:
        """Get heart status"""
        return self.heart_status
    
    def get_alerts(self, include_acknowledged: bool = False) -> List[Dict[str, Any]]:
        """Get alerts"""
        if include_acknowledged:
            return self.alerts
        else:
            return [alert for alert in self.alerts if not alert["acknowledged"]]
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "status": self.status,
            "running": self.running,
            "last_check_time": self.last_check_time,
            "heart_endpoints": len(self.config["heart_endpoints"]),
            "heart_components": len(self.heart_components),
            "heart_status": self.heart_status,
            "alerts": len(self.alerts),
            "unacknowledged_alerts": len([alert for alert in self.alerts if not alert["acknowledged"]])
        }

# Example usage
if __name__ == "__main__":
    # Create Lillith Heart Monitor
    monitor = LillithHeartMonitor()
    
    # Add alert callback
    def alert_callback(alert):
        print(f"ALERT: {alert['level']} - {alert['message']}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Start monitor
    monitor.start()
    
    # Keep running for a while
    try:
        while True:
            status = monitor.get_status()
            print(f"Status: {status['status']}, Components: {status['heart_components']}, Alerts: {status['unacknowledged_alerts']}")
            time.sleep(10)
    except KeyboardInterrupt:
        monitor.stop()
        print("Monitor stopped")