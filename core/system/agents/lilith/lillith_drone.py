# Systems/engine/viren_drone.py
# Purpose: Drone service for Viren to handle distributed tasks

import os
import sys
import time
import json
import logging
import threading
import requests
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("viren_drone")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/viren_drone.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class VirenDrone:
    """
    Drone service for Viren to handle distributed tasks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the drone service."""
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "drone_config.json")
        self.config = self._load_config()
        self.running = False
        self.threads = []
        
        # Initialize drone ID
        self.drone_id = self.config.get("drone_id", f"drone-{int(time.time())}")
        
        logger.info(f"Viren drone {self.drone_id} initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the drone configuration."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded drone configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default drone configuration."""
        return {
            "drone_id": f"drone-{int(time.time())}",
            "heartbeat_interval": 60,
            "master_url": "http://localhost:8080",
            "tasks": [
                {
                    "name": "heartbeat",
                    "enabled": True,
                    "interval": 60
                },
                {
                    "name": "status_report",
                    "enabled": True,
                    "interval": 300
                }
            ]
        }
    
    def start(self) -> bool:
        """
        Start the drone service.
        
        Returns:
            True if successful, False otherwise
        """
        if self.running:
            logger.warning("Drone service already running")
            return True
        
        self.running = True
        
        # Start task threads
        for task in self.config.get("tasks", []):
            if task.get("enabled", False):
                thread = threading.Thread(
                    target=self._run_task,
                    args=(task["name"], task.get("interval", 60)),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
                logger.info(f"Started task thread: {task['name']}")
        
        # Register with master
        self._register_with_master()
        
        logger.info(f"Viren drone {self.drone_id} started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the drone service.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.running:
            logger.warning("Drone service not running")
            return True
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Unregister from master
        self._unregister_from_master()
        
        logger.info(f"Viren drone {self.drone_id} stopped")
        return True
    
    def _run_task(self, task_name: str, interval: int) -> None:
        """
        Run a task at regular intervals.
        
        Args:
            task_name: Name of the task to run
            interval: Interval in seconds
        """
        while self.running:
            try:
                if task_name == "heartbeat":
                    self._send_heartbeat()
                elif task_name == "status_report":
                    self._send_status_report()
                else:
                    logger.warning(f"Unknown task: {task_name}")
            except Exception as e:
                logger.error(f"Error running task {task_name}: {e}")
            
            # Sleep for the interval
            time.sleep(interval)
    
    def _register_with_master(self) -> bool:
        """
        Register the drone with the master.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            master_url = self.config.get("master_url")
            if not master_url:
                logger.warning("Master URL not configured, skipping registration")
                return False
            
            response = requests.post(
                f"{master_url}/register",
                json={
                    "drone_id": self.drone_id,
                    "capabilities": ["heartbeat", "status_report"]
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Registered with master at {master_url}")
                return True
            else:
                logger.warning(f"Failed to register with master: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error registering with master: {e}")
            return False
    
    def _unregister_from_master(self) -> bool:
        """
        Unregister the drone from the master.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            master_url = self.config.get("master_url")
            if not master_url:
                logger.warning("Master URL not configured, skipping unregistration")
                return False
            
            response = requests.post(
                f"{master_url}/unregister",
                json={
                    "drone_id": self.drone_id
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Unregistered from master at {master_url}")
                return True
            else:
                logger.warning(f"Failed to unregister from master: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error unregistering from master: {e}")
            return False
    
    def _send_heartbeat(self) -> bool:
        """
        Send a heartbeat to the master.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            master_url = self.config.get("master_url")
            if not master_url:
                logger.debug("Master URL not configured, skipping heartbeat")
                return False
            
            response = requests.post(
                f"{master_url}/heartbeat",
                json={
                    "drone_id": self.drone_id,
                    "timestamp": time.time()
                }
            )
            
            if response.status_code == 200:
                logger.debug(f"Sent heartbeat to master at {master_url}")
                return True
            else:
                logger.warning(f"Failed to send heartbeat to master: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending heartbeat to master: {e}")
            return False
    
    def _send_status_report(self) -> bool:
        """
        Send a status report to the master.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            master_url = self.config.get("master_url")
            if not master_url:
                logger.debug("Master URL not configured, skipping status report")
                return False
            
            # Collect system status
            import psutil
            
            status = {
                "drone_id": self.drone_id,
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
            
            response = requests.post(
                f"{master_url}/status",
                json=status
            )
            
            if response.status_code == 200:
                logger.info(f"Sent status report to master at {master_url}")
                return True
            else:
                logger.warning(f"Failed to send status report to master: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending status report to master: {e}")
            return False

# Create a singleton instance
viren_drone = VirenDrone()

# Example usage
if __name__ == "__main__":
    # Start the drone service
    viren_drone.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop the drone service
        viren_drone.stop()