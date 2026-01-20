#!/usr/bin/env python3
"""
Blockchain Relay for Cloud Viren
Serves as a blockchain relay node when the system is idle
"""

import os
import json
import time
import logging
import threading
import socket
import hashlib
import base64
import queue
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger("VirenBlockchainRelay")

class BlockchainRelay:
    """
    Blockchain relay for Cloud Viren
    Acts as a relay node for the Nexus blockchain when idle
    """
    
    def __init__(self, node_endpoint: str = None, idle_threshold: int = 1800, config_path: str = None):
        """Initialize the blockchain relay"""
        self.node_endpoint = node_endpoint
        self.idle_threshold = idle_threshold  # 30 minutes by default
        self.config_path = config_path or os.path.join("config", "blockchain_config.json")
        self.config = self._load_config()
        self.running = False
        self.relay_thread = None
        self.last_activity = time.time()
        self.relay_status = "inactive"
        self.transaction_queue = queue.Queue()
        self.processed_transactions = []
        self.max_history = 100
        self.peers = set()
        self.node_id = self._generate_node_id()
        
        logger.info(f"Blockchain relay initialized with node ID: {self.node_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "node_endpoint": self.node_endpoint or "https://relay.nexus-blockchain.io",
            "idle_threshold": self.idle_threshold,
            "relay_port": 9876,
            "max_connections": 50,
            "heartbeat_interval": 60,  # seconds
            "transaction_limit": 1000,  # transactions per minute
            "bootstrap_nodes": [
                "relay1.nexus-blockchain.io:9876",
                "relay2.nexus-blockchain.io:9876",
                "relay3.nexus-blockchain.io:9876"
            ],
            "allowed_transaction_types": ["data", "diagnostic", "update"],
            "blockchain_sync": {
                "enabled": True,
                "interval": 3600,  # 1 hour
                "max_blocks": 1000
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
                    
                    logger.info("Blockchain configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading blockchain configuration: {e}")
        
        logger.info("Using default blockchain configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Blockchain configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving blockchain configuration: {e}")
            return False
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID"""
        # Use hostname, MAC address, and timestamp to create a unique ID
        hostname = socket.gethostname()
        try:
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0, 2*6, 2)][::-1])
        except:
            mac = "00:00:00:00:00:00"
        
        timestamp = str(time.time())
        
        # Create a hash of these components
        node_id_raw = hashlib.sha256(f"{hostname}:{mac}:{timestamp}".encode()).digest()
        node_id = base64.b32encode(node_id_raw).decode()[:16]
        
        return f"VIREN-{node_id}"
    
    def start(self) -> bool:
        """Start the blockchain relay"""
        if self.running:
            logger.warning("Blockchain relay is already running")
            return False
        
        logger.info("Starting blockchain relay")
        self.running = True
        self.relay_status = "starting"
        
        # Start relay thread
        self.relay_thread = threading.Thread(target=self._relay_loop)
        self.relay_thread.daemon = True
        self.relay_thread.start()
        
        return True
    
    def stop(self) -> bool:
        """Stop the blockchain relay"""
        if not self.running:
            logger.warning("Blockchain relay is not running")
            return False
        
        logger.info("Stopping blockchain relay")
        self.running = False
        self.relay_status = "stopping"
        
        # Wait for relay thread to finish
        if self.relay_thread and self.relay_thread.is_alive():
            self.relay_thread.join(timeout=5)
        
        self.relay_status = "inactive"
        return True
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def is_idle(self) -> bool:
        """Check if the system is idle"""
        return (time.time() - self.last_activity) > self.config["idle_threshold"]
    
    def _relay_loop(self) -> None:
        """Main relay loop"""
        logger.info("Relay loop started")
        self.relay_status = "active"
        
        # Connect to bootstrap nodes
        self._connect_to_bootstrap_nodes()
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # Start listening for connections
        listen_thread = threading.Thread(target=self._listen_for_connections)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Process transactions
        while self.running:
            try:
                # Check if we should be active
                if not self.is_idle():
                    logger.debug("System not idle, pausing relay operations")
                    self.relay_status = "paused"
                    time.sleep(60)  # Check again in a minute
                    continue
                
                self.relay_status = "active"
                
                # Process transactions from queue
                self._process_transaction_queue()
                
                # Sync with blockchain periodically
                if self.config["blockchain_sync"]["enabled"]:
                    self._sync_with_blockchain()
                
                # Sleep briefly
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in relay loop: {e}")
                time.sleep(5)  # Wait before retrying
        
        logger.info("Relay loop stopped")
    
    def _connect_to_bootstrap_nodes(self) -> None:
        """Connect to bootstrap nodes"""
        logger.info("Connecting to bootstrap nodes")
        
        for node in self.config["bootstrap_nodes"]:
            try:
                # In a real implementation, this would establish actual connections
                logger.info(f"Connected to bootstrap node: {node}")
                self.peers.add(node)
            except Exception as e:
                logger.error(f"Failed to connect to bootstrap node {node}: {e}")
    
    def _heartbeat_loop(self) -> None:
        """Send heartbeat to peers periodically"""
        logger.info("Heartbeat loop started")
        
        while self.running:
            try:
                if self.relay_status == "active":
                    # Send heartbeat to all peers
                    for peer in list(self.peers):
                        try:
                            # In a real implementation, this would send actual heartbeat messages
                            logger.debug(f"Sent heartbeat to peer: {peer}")
                        except Exception as e:
                            logger.error(f"Failed to send heartbeat to peer {peer}: {e}")
                            self.peers.remove(peer)
                
                # Sleep until next heartbeat
                time.sleep(self.config["heartbeat_interval"])
            
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)  # Wait before
