# Systems/engine/error_recovery.py
# Purpose: Error recovery and self-healing mechanisms for Viren

import os
import sys
import time
import json
import logging
import threading
import subprocess
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logger = logging.getLogger("error_recovery")
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
handler = logging.FileHandler("logs/error_recovery.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ErrorRecovery:
    """
    Error recovery and self-healing mechanisms for Viren.
    """
    
    def __init__(self):
        """Initialize the error recovery system."""
        self.running = False
        self.critical_services = [
            "memory",
            "heart",
            "consciousness",
            "subconscious"
        ]
        self.service_health = {}
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 300  # 5 minutes
        
        # Recovery handlers
        self.recovery_handlers = {
            "service_down": self._recover_service,
            "memory_corruption": self._recover_memory,
            "model_failure": self._recover_model,
            "connection_lost": self._recover_connection
        }
        
        # Health check callbacks
        self.health_check_callbacks = []
    
    def start(self):
        """Start the error recovery system."""
        if self.running:
            return
            
        self.running = True
        
        # Start health monitoring
        threading.Thread(target=self._monitor_health, daemon=True).start()
        
        # Start recovery monitor
        threading.Thread(target=self._monitor_recovery_attempts, daemon=True).start()
        
        logger.info("Error recovery system started")
    
    def stop(self):
        """Stop the error recovery system."""
        if not self.running:
            return
            
        self.running = False
        logger.info("Error recovery system stopped")
    
    def register_health_check(self, callback: Callable[[], Dict[str, Any]]):
        """
        Register a health check callback.
        
        Args:
            callback: Function that returns health status
        """
        self.health_check_callbacks.append(callback)
        logger.info(f"Registered health check callback: {callback.__name__}")
    
    def report_error(self, error_type: str, component: str, details: Dict[str, Any]) -> bool:
        """
        Report an error for recovery.
        
        Args:
            error_type: Type of error
            component: Component that experienced the error
            details: Error details
            
        Returns:
            True if recovery was attempted, False otherwise
        """
        logger.warning(f"Error reported: {error_type} in {component}")
        
        # Check if we have a recovery handler for this error type
        if error_type not in self.recovery_handlers:
            logger.error(f"No recovery handler for error type: {error_type}")
            return False
        
        # Check if we've exceeded recovery attempts
        recovery_key = f"{error_type}:{component}"
        if recovery_key in self.recovery_attempts:
            attempts, last_time = self.recovery_attempts[recovery_key]
            
            # Check if we're in cooldown
            if time.time() - last_time < self.recovery_cooldown:
                # Check if we've exceeded max attempts
                if attempts >= self.max_recovery_attempts:
                    logger.error(f"Exceeded max recovery attempts for {recovery_key}")
                    return False
        
        # Attempt recovery
        success = self.recovery_handlers[error_type](component, details)
        
        # Update recovery attempts
        if recovery_key in self.recovery_attempts:
            attempts, _ = self.recovery_attempts[recovery_key]
            self.recovery_attempts[recovery_key] = (attempts + 1, time.time())
        else:
            self.recovery_attempts[recovery_key] = (1, time.time())
        
        return success
    
    def _monitor_health(self):
        """Monitor system health."""
        while self.running:
            try:
                # Collect health status from all callbacks
                health_status = {}
                for callback in self.health_check_callbacks:
                    try:
                        status = callback()
                        health_status.update(status)
                    except Exception as e:
                        logger.error(f"Error in health check callback {callback.__name__}: {e}")
                
                # Check critical services
                for service in self.critical_services:
                    if service in health_status:
                        if health_status[service].get("status") == "down":
                            # Attempt recovery
                            self.report_error("service_down", service, health_status[service])
                
                # Update service health
                self.service_health = health_status
            except Exception as e:
                logger.error(f"Error monitoring health: {e}")
            
            time.sleep(60)  # Check every minute
    
    def _monitor_recovery_attempts(self):
        """Monitor and reset recovery attempts."""
        while self.running:
            try:
                current_time = time.time()
                for key, (attempts, last_time) in list(self.recovery_attempts.items()):
                    # Reset attempts after cooldown period
                    if current_time - last_time > self.recovery_cooldown * 2:
                        del self.recovery_attempts[key]
                        logger.info(f"Reset recovery attempts for {key}")
            except Exception as e:
                logger.error(f"Error monitoring recovery attempts: {e}")
            
            time.sleep(300)  # Check every 5 minutes
    
    def _recover_service(self, service: str, details: Dict[str, Any]) -> bool:
        """
        Recover a down service.
        
        Args:
            service: Service name
            details: Error details
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to recover service: {service}")
        
        try:
            # Check if service is running
            if self._is_service_running(service):
                logger.info(f"Service {service} is already running")
                return True
            
            # Start the service
            launch_script = os.path.join("Systems", service, f"launch_{service}.py")
            if os.path.exists(launch_script):
                subprocess.Popen([sys.executable, launch_script])
                logger.info(f"Started service: {service}")
                return True
            else:
                logger.error(f"Service launch script not found: {launch_script}")
                return False
        except Exception as e:
            logger.error(f"Error recovering service {service}: {e}")
            return False
    
    def _recover_memory(self, component: str, details: Dict[str, Any]) -> bool:
        """
        Recover from memory corruption.
        
        Args:
            component: Component with corrupted memory
            details: Error details
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to recover memory for component: {component}")
        
        try:
            # Check if we have a backup
            memory_path = details.get("memory_path")
            if not memory_path or not os.path.exists(memory_path):
                logger.error(f"Memory path not found: {memory_path}")
                return False
            
            # Check for backup
            backup_path = f"{memory_path}.backup"
            if not os.path.exists(backup_path):
                logger.error(f"Memory backup not found: {backup_path}")
                return False
            
            # Restore from backup
            import shutil
            shutil.copy2(backup_path, memory_path)
            logger.info(f"Restored memory from backup for {component}")
            return True
        except Exception as e:
            logger.error(f"Error recovering memory for {component}: {e}")
            return False
    
    def _recover_model(self, model_name: str, details: Dict[str, Any]) -> bool:
        """
        Recover from model failure.
        
        Args:
            model_name: Name of the failed model
            details: Error details
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to recover model: {model_name}")
        
        try:
            # Get fallback model
            fallback_model = details.get("fallback_model")
            if not fallback_model:
                logger.error(f"No fallback model specified for {model_name}")
                return False
            
            # Update model configuration
            try:
                from config.model_config import update_model_config
                update_model_config({model_name: fallback_model})
                logger.info(f"Updated model configuration to use fallback model for {model_name}")
                return True
            except Exception as e:
                logger.error(f"Error updating model configuration: {e}")
                return False
        except Exception as e:
            logger.error(f"Error recovering model {model_name}: {e}")
            return False
    
    def _recover_connection(self, component: str, details: Dict[str, Any]) -> bool:
        """
        Recover from connection loss.
        
        Args:
            component: Component that lost connection
            details: Error details
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to recover connection for component: {component}")
        
        try:
            # Get connection details
            connection_type = details.get("connection_type")
            if not connection_type:
                logger.error(f"No connection type specified for {component}")
                return False
            
            if connection_type == "service":
                # Try to reconnect to service
                service_name = details.get("service_name")
                if not service_name:
                    logger.error(f"No service name specified for {component}")
                    return False
                
                # Import service discovery
                try:
                    from Systems.engine.core.service_discovery import service_discovery
                    
                    # Restart service discovery if needed
                    if not service_discovery.running:
                        service_discovery.start_discovery()
                    
                    # Service will be rediscovered automatically
                    logger.info(f"Restarted service discovery to reconnect {component} to {service_name}")
                    return True
                except Exception as e:
                    logger.error(f"Error restarting service discovery: {e}")
                    return False
            elif connection_type == "database":
                # Try to reconnect to database
                # This is a simplified example
                logger.info(f"Database reconnection not implemented yet for {component}")
                return False
            else:
                logger.error(f"Unknown connection type: {connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error recovering connection for {component}: {e}")
            return False
    
    def _is_service_running(self, service: str) -> bool:
        """
        Check if a service is running.
        
        Args:
            service: Service name
            
        Returns:
            True if the service is running, False otherwise
        """
        try:
            # Get list of running Python processes
            if sys.platform == "win32":
                cmd = ["wmic", "process", "where", "name='python.exe'", "get", "commandline", "/format:csv"]
            else:
                cmd = ["ps", "-ef", "|", "grep", "python"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            
            # Check if service is in the output
            return f"launch_{service}.py" in output
        except Exception as e:
            logger.error(f"Error checking if service {service} is running: {e}")
            return False

# Create singleton instance
error_recovery = ErrorRecovery()

# Example usage
if __name__ == "__main__":
    # Start error recovery
    error_recovery.start()
    
    # Example health check callback
    def check_service_health():
        return {
            "memory": {"status": "up", "memory_usage": 100},
            "heart": {"status": "up", "pulse_rate": 60},
            "consciousness": {"status": "up", "awareness_level": 0.95}
        }
    
    # Register health check
    error_recovery.register_health_check(check_service_health)
    
    # Example error report
    error_recovery.report_error("service_down", "subconscious", {"reason": "crashed"})
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        error_recovery.stop()