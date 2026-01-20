#!/usr/bin/env python3
"""
Cloud Connection for Viren
Handles automatic connection between desktop Viren and Cloud Viren
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import requests
import socket
import ssl
import base64
import hashlib
import platform
import asyncio
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VirenCloudConnection")

# Import Modal (if available)
try:
    import modal
    modal_available = True
except ImportError:
    logger.warning("Modal package not found. Install with: pip install modal")
    modal_available = False

class CloudConnection:
    """
    Cloud Connection for Viren
    Handles automatic connection between desktop Viren and Cloud Viren
    """

    def __init__(self, config_path: str = None):
        """Initialize the cloud connection"""
        self.config_path = config_path or os.path.join("config", "cloud_connection_config.json")
        self.config = self._load_config()
        self.running = False
        self.connection_thread = None
        self.heartbeat_thread = None
        self.status = "disconnected"
        self.last_connection_time = 0
        self.last_heartbeat_time = 0
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.backoff_time = 5  # seconds
        self.session_id = None
        self.device_id = self._get_or_create_device_id()
        self.cloud_capabilities = {}
        self.sync_queue = []
        self.sync_lock = threading.Lock()
        self.modal_models = {}

        logger.info(f"Cloud Connection initialized with device ID: {self.device_id}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "cloud_endpoint": "https://api.viren-cloud.com/v1",
            "api_key": "",
            "connection_interval": 60,  # seconds
            "heartbeat_interval": 30,  # seconds
            "auto_connect": True,
            "auto_reconnect": True,
            "secure_connection": True,
            "sync_models": True,
            "sync_knowledge": True,
            "sync_diagnostics": True,
            "sync_interval": 300,  # seconds
            "max_sync_queue_size": 1000,
            "device_name": socket.gethostname(),
            "modal": {
                "app_name": "aethereal-nexus",
                "namespace": "aethereal-nexus",
                "models": {
                    "1B": "gemma_1b",
                    "3B": "gemma_3b",
                    "7B": "llama_7b"
                }
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

                    logger.info("Cloud connection configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading cloud connection configuration: {e}")

        logger.info("Using default cloud connection configuration")
        return default_config

    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Cloud connection configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving cloud connection configuration: {e}")
            return False

    def _get_or_create_device_id(self) -> str:
        """Get or create a unique device ID"""
        device_id_path = os.path.join("config", "device_id.txt")

        if os.path.exists(device_id_path):
            try:
                with open(device_id_path, 'r') as f:
                    device_id = f.read().strip()
                    if device_id:
                        return device_id
            except Exception as e:
                logger.error(f"Error reading device ID: {e}")

        # Create a new device ID
        try:
            # Use hardware info to create a stable ID
            machine_id = self._get_machine_id()
            device_id = f"VIREN-{machine_id}-{uuid.uuid4().hex[:8]}"

            # Save device ID
            os.makedirs(os.path.dirname(device_id_path), exist_ok=True)
            with open(device_id_path, 'w') as f:
                f.write(device_id)

            return device_id
        except Exception as e:
            logger.error(f"Error creating device ID: {e}")
            return f"VIREN-UNKNOWN-{uuid.uuid4().hex[:8]}"

    def _get_machine_id(self) -> str:
        """Get a stable machine ID based on hardware"""
        try:
            # Collect hardware information
            hardware_info = {
                "hostname": socket.gethostname(),
                "platform": sys.platform,
                "processor": platform.processor() if hasattr(platform, 'processor') else "unknown",
                "mac_address": self._get_mac_address()
            }

            # Create a hash of the hardware info
            hardware_str = json.dumps(hardware_info, sort_keys=True)
            machine_id = hashlib.md5(hardware_str.encode()).hexdigest()[:12]

            return machine_id
        except Exception as e:
            logger.error(f"Error getting machine ID: {e}")
            return uuid.uuid4().hex[:12]

    def _get_mac_address(self) -> str:
        """Get MAC address of the first network interface"""
        try:
            import uuid
            return ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                           for elements in range(0, 2*6, 2)][::-1])
        except Exception as e:
            logger.error(f"Error getting MAC address: {e}")
            return "00:00:00:00:00:00"

    def initialize_modal_connection(self) -> bool:
        """Initialize connection to Modal-deployed models"""
        if not modal_available:
            logger.error("Modal package not available, cannot connect to Modal models")
            return False

        try:
            app_name = self.config["modal"]["app_name"]
            namespace = self.config["modal"]["namespace"]

            # Connect to deployed models
            self.modal_models = {}
            for size, function_name in self.config["modal"]["models"].items():
                try:
                    self.modal_models[size] = modal.Function.lookup(app_name, function_name, namespace=namespace)
                    logger.info(f"Connected to Modal model {size} ({function_name})")
                except Exception as e:
                    logger.error(f"Error connecting to Modal model {size} ({function_name}): {e}")

            if self.modal_models:
                logger.info(f"Connected to {len(self.modal_models)} Modal-deployed models")
                return True
            else:
                logger.warning("No Modal models connected")
                return False
        except Exception as e:
            logger.error(f"Error initializing Modal connection: {e}")
            return False

    def start(self) -> bool:
        """Start the cloud connection"""
        if self.running:
            logger.warning("Cloud connection is already running")
            return False

        logger.info("Starting cloud connection")
        self.running = True
        self.status = "connecting"

        # Initialize Modal connection
        if modal_available:
            self.initialize_modal_connection()

        # Start connection thread
        self.connection_thread = threading.Thread(target=self._connection_loop)
        self.connection_thread.daemon = True
        self.connection_thread.start()

        return True

    def stop(self) -> bool:
        """Stop the cloud connection"""
        if not self.running:
            logger.warning("Cloud connection is not running")
            return False

        logger.info("Stopping cloud connection")
        self.running = False
        self.status = "disconnecting"

        # Wait for connection thread to finish
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=5)

        # Wait for heartbeat thread to finish
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)

        self.status = "disconnected"
        logger.info("Cloud connection stopped")
        return True

    def _connection_loop(self) -> None:
        """Main connection loop"""
        logger.info("Starting connection loop")

        while self.running:
            try:
                # Check if we should connect
                if self.config["auto_connect"] and self.status in ["disconnected", "error"]:
                    self._connect_to_cloud()

                # Sleep until next connection attempt
                time.sleep(self.config["connection_interval"])

            except Exception as e:
                logger.error(f"Error in connection loop: {e}")
                self.status = "error"
                time.sleep(self.backoff_time)
                self.backoff_time = min(self.backoff_time * 2, 300)  # Max 5 minutes

    def _connect_to_cloud(self) -> bool:
        """Connect to Cloud Viren"""
        logger.info("Connecting to Cloud Viren")
        self.status = "connecting"
        self.connection_attempts += 1

        try:
            # Prepare connection request
            endpoint = f"{self.config['cloud_endpoint']}/connect"
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.config["api_key"],
                "User-Agent": f"Viren-Desktop/{self._get_version()}"
            }

            payload = {
                "device_id": self.device_id,
                "device_name": self.config["device_name"],
                "platform": sys.platform,
                "version": self._get_version(),
                "capabilities": self._get_capabilities()
            }

            # Send connection request
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                # Connection successful
                result = response.json()
                self.session_id = result.get("session_id")
                self.cloud_capabilities = result.get("capabilities", {})
                self.status = "connected"
                self.last_connection_time = time.time()
                self.connection_attempts = 0
                self.backoff_time = 5  # Reset backoff time

                logger.info(f"Connected to Cloud Viren with session ID: {self.session_id}")

                # Start heartbeat thread
                if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
                    self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
                    self.heartbeat_thread.daemon = True
                    self.heartbeat_thread.start()

                # Sync with cloud
                self._sync_with_cloud()

                return True
            else:
                # Connection failed
                logger.error(f"Failed to connect to Cloud Viren: {response.status_code} - {response.text}")
                self.status = "error"

                # Check if we should retry
                if self.connection_attempts >= self.max_connection_attempts:
                    logger.error(f"Max connection attempts reached ({self.max_connection_attempts}), giving up")
                    self.status = "disconnected"
                    return False

                return False

        except Exception as e:
            logger.error(f"Error connecting to Cloud Viren: {e}")
            self.status = "error"
            return False

    def _heartbeat_loop(self) -> None:
        """Send heartbeat to Cloud Viren periodically"""
        logger.info("Starting heartbeat loop")

        while self.running and self.status == "connected":
            try:
                # Send heartbeat
                self._send_heartbeat()

                # Sleep until next heartbeat
                time.sleep(self.config["heartbeat_interval"])

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)  # Wait before retrying

    def _send_heartbeat(self) -> bool:
        """Send heartbeat to Cloud Viren"""
        try:
            # Prepare heartbeat request
            endpoint = f"{self.config['cloud_endpoint']}/heartbeat"
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.config["api_key"],
                "X-Session-ID": self.session_id,
                "User-Agent": f"Viren-Desktop/{self._get_version()}"
            }

            payload = {
                "device_id": self.device_id,
                "session_id": self.session_id,
                "timestamp": time.time(),
                "status": "active"
            }

            # Send heartbeat request
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                # Heartbeat successful
                self.last_heartbeat_time = time.time()
                logger.debug("Heartbeat sent successfully")
                return True
            else:
                # Heartbeat failed
                logger.warning(f"Failed to send heartbeat: {response.status_code} - {response.text}")

                # Check if session is invalid
                if response.status_code == 401:
                    logger.warning("Session invalid, reconnecting")
                    self.status = "disconnected"
                    self._connect_to_cloud()

                return False

        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False

    def _sync_with_cloud(self) -> bool:
        """Sync with Cloud Viren"""
        logger.info("Syncing with Cloud Viren")

        try:
            # Process sync queue
            with self.sync_lock:
                if not self.sync_queue:
                    logger.info("Sync queue is empty")
                    return True

                # Prepare sync request
                endpoint = f"{self.config['cloud_endpoint']}/sync"
                headers = {
                    "Content-Type": "application/json",
                    "X-API-Key": self.config["api_key"],
                    "X-Session-ID": self.session_id,
                    "User-Agent": f"Viren-Desktop/{self._get_version()}"
                }

                # Process sync items in batches
                batch_size = 50
                for i in range(0, len(self.sync_queue), batch_size):
                    batch = self.sync_queue[i:i+batch_size]

                    payload = {
                        "device_id": self.device_id,
                        "session_id": self.session_id,
                        "timestamp": time.time(),
                        "items": batch
                    }

                    # Send sync request
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:
                        # Sync successful
                        logger.info(f"Synced {len(batch)} items successfully")
                    else:
                        # Sync failed
                        logger.error(f"Failed to sync items: {response.status_code} - {response.text}")
                        return False

                # Clear sync queue
                self.sync_queue = []

            return True

        except Exception as e:
            logger.error(f"Error syncing with Cloud Viren: {e}")
            return False

    def queue_sync_item(self, item_type: str, data: Dict[str, Any]) -> bool:
        """Queue an item for syncing with Cloud Viren"""
        try:
            # Create sync item
            sync_item = {
                "type": item_type,
                "timestamp": time.time(),
                "data": data
            }

            # Add to sync queue
            with self.sync_lock:
                self.sync_queue.append(sync_item)

                # Check if queue is too large
                if len(self.sync_queue) > self.config["max_sync_queue_size"]:
                    # Remove oldest items
                    excess = len(self.sync_queue) - self.config["max_sync_queue_size"]
                    self.sync_queue = self.sync_queue[excess:]
                    logger.warning(f"Sync queue too large, removed {excess} oldest items")

            # Trigger sync if connected
            if self.status == "connected":
                threading.Thread(target=self._sync_with_cloud).start()

            return True

        except Exception as e:
            logger.error(f"Error queuing sync item: {e}")
            return False

    async def generate_text(self, prompt: str, model_size: str = "3B",
                          max_tokens: int = 1024, temperature: float = 0.7,
                          top_p: float = 0.9) -> Dict[str, Any]:
        """Generate text using cloud-deployed models"""
        if not modal_available:
            return {"error": "Modal package not available", "output": "Error: Modal package not available"}

        if not self.modal_models:
            self.initialize_modal_connection()

        try:
            if model_size not in self.modal_models:
                raise ValueError(f"Unsupported model size: {model_size}")

            # Get the appropriate model
            model = self.modal_models[model_size]

            # Generate text
            result = await model.remote(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Queue sync item for tracking
            self.queue_sync_item("model_inference", {
                "model_size": model_size,
                "prompt_length": len(prompt),
                "output_length": len(result.get("output", "")),
                "temperature": temperature,
                "top_p": top_p
            })

            return result
        except Exception as e:
            logger.error(f"Error generating text with model {model_size}: {e}")
            return {
                "error": str(e),
                "output": f"Error: {str(e)}"
            }

    def _get_version(self) -> str:
        """Get Viren version"""
        try:
            version_path = os.path.join("config", "version.txt")
            if os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    return f.read().strip()
            return "1.0.0"  # Default version
        except Exception as e:
            logger.error(f"Error getting version: {e}")
            return "1.0.0"

    def _get_capabilities(self) -> Dict[str, Any]:
        """Get desktop Viren capabilities"""
        return {
            "models": self._get_available_models(),
            "diagnostics": True,
            "research": True,
            "blockchain_relay": True,
            "memory": self._get_available_memory(),
            "disk": self._get_available_disk(),
            "cpu": self._get_available_cpu(),
            "modal_models": list(self.modal_models.keys()) if hasattr(self, 'modal_models') else []
        }

    def _get_available_models(self) -> List[str]:
        """Get available models"""
        try:
            models_dir = os.path.join("models")
            if os.path.exists(models_dir):
                return [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    def _get_available_memory(self) -> Dict[str, Any]:
        """Get available memory"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent
            }
        except Exception as e:
            logger.error(f"Error getting available memory: {e}")
            return {"total": 0, "available": 0, "percent": 0}

    def _get_available_disk(self) -> Dict[str, Any]:
        """Get available disk space"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent
            }
        except Exception as e:
            logger.error(f"Error getting available disk space: {e}")
            return {"total": 0, "free": 0, "percent": 0}

    def _get_available_cpu(self) -> Dict[str, Any]:
        """Get available CPU"""
        try:
            import psutil
            return {
                "count": psutil.cpu_count(logical=True),
                "physical_count": psutil.cpu_count(logical=False),
                "percent": psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            logger.error(f"Error getting available CPU: {e}")
            return {"count": 0, "physical_count": 0, "percent": 0}

    def get_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            "status": self.status,
            "running": self.running,
            "device_id": self.device_id,
            "session_id": self.session_id,
            "last_connection_time": self.last_connection_time,
            "last_heartbeat_time": self.last_heartbeat_time,
            "connection_attempts": self.connection_attempts,
            "sync_queue_size": len(self.sync_queue),
            "cloud_capabilities": self.cloud_capabilities,
            "modal_models": list(self.modal_models.keys()) if hasattr(self, 'modal_models') else []
        }

    def is_connected(self) -> bool:
        """Check if connected to Cloud Viren"""
        return self.status == "connected"

# Example usage
if __name__ == "__main__":
    # Create cloud connection
    connection = CloudConnection()

    # Start connection
    connection.start()

    # Test Modal connection
    if modal_available:
        async def test_modal():
            print("Testing Modal connection...")
            result = await connection.generate_text(
                "What is Aethereal Nexus?",
                model_size="3B"
            )
            print(f"Model output: {result.get('output', 'No output')}")

        asyncio.run(test_modal())

    # Keep running for a while
    try:
        while True:
            status = connection.get_status()
            print(f"Status: {status['status']}, Modal models: {status.get('modal_models', [])}")
            time.sleep(10)
    except KeyboardInterrupt:
        connection.stop()
        print("Connection stopped")
