#!/usr/bin/env python3
"""
Model Collective for Viren Cloud - Enables communication between multiple LLMs
"""

import os
import sys
import time
import json
import threading
import logging
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelCollective")

class ModelNode:
    """Represents a single LLM in the collective"""
    
    def __init__(self, model_id: str, model_type: str, model_instance: Any, 
                 tokenizer: Any = None, role: str = "general"):
        self.model_id = model_id
        self.model_type = model_type
        self.model = model_instance
        self.tokenizer = tokenizer
        self.role = role
        self.message_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.collective = None  # Will be set by the collective
        
    def start(self):
        """Start the model node processing loop"""
        self.is_running = True
        self.thread = threading.Thread(target=self._process_messages)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Model node {self.model_id} started with role: {self.role}")
        
    def stop(self):
        """Stop the model node"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info(f"Model node {self.model_id} stopped")
        
    def send_message(self, message: Dict[str, Any]):
        """Send a message to this model node"""
        self.message_queue.put(message)
        
    def _process_messages(self):
        """Process incoming messages"""
        while self.is_running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(block=False)
                    self._handle_message(message)
                    self.message_queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing message in {self.model_id}: {str(e)}")
                time.sleep(1)
                
    def _handle_message(self, message: Dict[str, Any]):
        """Handle an incoming message"""
        msg_type = message.get("type", "")
        content = message.get("content", "")
        sender = message.get("sender", "")
        
        if msg_type == "query":
            # Generate a response to the query
            response = self.generate_response(content)
            
            # Send the response back to the collective
            if self.collective:
                self.collective.broadcast_message({
                    "type": "response",
                    "sender": self.model_id,
                    "recipient": sender,
                    "content": response,
                    "role": self.role
                })
                
        elif msg_type == "response":
            # This is a response to a query we sent
            # In a real implementation, we might do something with this
            pass
            
        elif msg_type == "broadcast":
            # This is a broadcast message for all models
            # We might want to update our internal state based on this
            pass
            
    def generate_response(self, prompt: str) -> str:
        """Generate a response using this model"""
        try:
            if self.model_type == "transformers":
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(inputs.input_ids, max_new_tokens=100)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            elif self.model_type == "llama_cpp":
                return self.model(prompt, max_tokens=100)["choices"][0]["text"]
                
            elif self.model_type == "langengine":
                # Assuming langengine has a similar API
                return self.model.generate(prompt, max_tokens=100)
                
            return f"Model {self.model_id} of type {self.model_type} cannot generate responses"
            
        except Exception as e:
            logger.error(f"Error generating response with {self.model_id}: {str(e)}")
            return f"Error generating response: {str(e)}"


class ModelCollective:
    """Manages a collective of LLMs that can communicate with each other"""
    
    def __init__(self):
        self.models = {}  # model_id -> ModelNode
        self.roles = {}   # role -> [model_ids]
        self.message_history = []
        self.max_history = 1000
        self.is_running = False
        
        # Try to import langengine
        try:
            import langengine
            self.langengine = langengine
            logger.info("Langengine successfully imported")
        except ImportError:
            logger.warning("Langengine not found. Some features may be limited.")
            self.langengine = None
            
    def add_model(self, model_node: ModelNode):
        """Add a model to the collective"""
        model_id = model_node.model_id
        role = model_node.role
        
        # Set the collective reference
        model_node.collective = self
        
        # Add to models dict
        self.models[model_id] = model_node
        
        # Add to roles dict
        if role not in self.roles:
            self.roles[role] = []
        self.roles[role].append(model_id)
        
        logger.info(f"Added model {model_id} with role {role} to the collective")
        
    def remove_model(self, model_id: str):
        """Remove a model from the collective"""
        if model_id in self.models:
            model = self.models[model_id]
            role = model.role
            
            # Stop the model
            model.stop()
            
            # Remove from models dict
            del self.models[model_id]
            
            # Remove from roles dict
            if role in self.roles and model_id in self.roles[role]:
                self.roles[role].remove(model_id)
                
            logger.info(f"Removed model {model_id} from the collective")
            
    def start(self):
        """Start the collective"""
        self.is_running = True
        
        # Start all model nodes
        for model_id, model in self.models.items():
            model.start()
            
        logger.info(f"Model collective started with {len(self.models)} models")
        
    def stop(self):
        """Stop the collective"""
        self.is_running = False
        
        # Stop all model nodes
        for model_id, model in self.models.items():
            model.stop()
            
        logger.info("Model collective stopped")
        
    def send_message_to_model(self, model_id: str, message: Dict[str, Any]):
        """Send a message to a specific model"""
        if model_id in self.models:
            self.models[model_id].send_message(message)
            self._add_to_history(message)
        else:
            logger.warning(f"Model {model_id} not found in collective")
            
    def send_message_to_role(self, role: str, message: Dict[str, Any]):
        """Send a message to all models with a specific role"""
        if role in self.roles:
            for model_id in self.roles[role]:
                self.models[model_id].send_message(message)
            self._add_to_history(message)
        else:
            logger.warning(f"Role {role} not found in collective")
            
    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all models"""
        for model_id, model in self.models.items():
            # Don't send the message back to the sender
            if message.get("sender") != model_id:
                model.send_message(message)
                
        self._add_to_history(message)
        
    def query(self, prompt: str, role: Optional[str] = None, 
              callback: Optional[Callable[[str, str], None]] = None) -> List[Dict[str, Any]]:
        """
        Query models in the collective and collect responses
        
        Args:
            prompt: The query text
            role: Optional role to target specific models
            callback: Optional callback function(model_id, response) for each response
            
        Returns:
            List of response dictionaries
        """
        query_id = f"query_{int(time.time())}"
        responses = []
        response_event = threading.Event()
        
        # Create a collector for responses
        def collect_response(message):
            if message.get("type") == "response" and message.get("query_id") == query_id:
                responses.append({
                    "model_id": message.get("sender"),
                    "role": message.get("role"),
                    "content": message.get("content")
                })
                
                # Call the callback if provided
                if callback:
                    callback(message.get("sender"), message.get("content"))
                    
                # If we've heard from all models, set the event
                if role:
                    if len(responses) >= len(self.roles.get(role, [])):
                        response_event.set()
                else:
                    if len(responses) >= len(self.models):
                        response_event.set()
        
        # Register a temporary listener for responses
        self._add_listener(collect_response)
        
        # Send the query
        message = {
            "type": "query",
            "sender": "user",
            "query_id": query_id,
            "content": prompt
        }
        
        if role:
            self.send_message_to_role(role, message)
        else:
            self.broadcast_message(message)
            
        # Wait for responses (with timeout)
        response_event.wait(timeout=30)
        
        # Remove the listener
        self._remove_listener(collect_response)
        
        return responses
    
    def _add_to_history(self, message: Dict[str, Any]):
        """Add a message to the history"""
        self.message_history.append({
            "timestamp": time.time(),
            "message": message
        })
        
        # Trim history if needed
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
            
    # Listener management for advanced use cases
    def _add_listener(self, listener_func):
        """Add a message listener"""
        pass  # Implement if needed
        
    def _remove_listener(self, listener_func):
        """Remove a message listener"""
        pass  # Implement if needed


# Create deployment models with specific roles
def create_deployment_monitor(collective: ModelCollective, model_instance, model_type="transformers", tokenizer=None):
    """Create a deployment monitor model"""
    monitor_prompt = """
    You are a deployment model for Viren, A next Generation AI Model developed by Aethereal AI Nexus.
    Your job is to monitor and confirm everything is running as it should and all services are deployed.
    Once everything deploys, You will download additional models to assist.
    For each database deployed, you will install one more sql Llama model to manage the weaviate databases
    and they should have the following prompt to assist them:
    "You are a deployment model for Viren, A next Generation AI Model developed by Aethereal AI Nexus.
    Your job is to monitor, index and maintain the database you are deployed with.
    Alert your primary model of any issues and log said issues"
    """
    
    # Create a specialized model node
    class DeploymentMonitorNode(ModelNode):
        def _handle_message(self, message):
            # Add specialized handling for deployment monitoring
            super()._handle_message(message)
            
            # Check services periodically
            if message.get("type") == "check_services":
                # Check services and report status
                services_status = self._check_services()
                
                # Broadcast the status
                if self.collective:
                    self.collective.broadcast_message({
                        "type": "service_status",
                        "sender": self.model_id,
                        "content": services_status
                    })
                    
        def _check_services(self):
            """Check the status of all services"""
            # In a real implementation, we would check actual services
            return {
                "weaviate": "running",
                "transformers": "running",
                "binary_protocol": "running",
                "api": "running",
                "ui": "running",
                "moshi_voice": "running"
            }
    
    # Create and add the monitor node
    monitor = DeploymentMonitorNode(
        model_id="deployment_monitor",
        model_type=model_type,
        model_instance=model_instance,
        tokenizer=tokenizer,
        role="monitor"
    )
    
    collective.add_model(monitor)
    return monitor


def create_database_manager(collective: ModelCollective, model_instance, db_name, 
                           model_type="transformers", tokenizer=None):
    """Create a database manager model"""
    db_prompt = f"""
    You are a deployment model for Viren, A next Generation AI Model developed by Aethereal AI Nexus.
    Your job is to monitor, index and maintain the {db_name} database you are deployed with.
    Alert your primary model of any issues and log said issues.
    """
    
    # Create a specialized model node
    class DatabaseManagerNode(ModelNode):
        def _handle_message(self, message):
            # Add specialized handling for database management
            super()._handle_message(message)
            
            # Handle database-specific queries
            if message.get("type") == "db_query" and message.get("db_name") == db_name:
                # Process the database query
                query = message.get("content", "")
                result = self._process_db_query(query)
                
                # Send the response
                if self.collective:
                    self.collective.broadcast_message({
                        "type": "db_result",
                        "sender": self.model_id,
                        "recipient": message.get("sender"),
                        "content": result,
                        "db_name": db_name
                    })
                    
        def _process_db_query(self, query):
            """Process a database query"""
            # In a real implementation, we would query the actual database
            return f"Processed query for {db_name}: {query}"
    
    # Create and add the database manager node
    manager = DatabaseManagerNode(
        model_id=f"db_manager_{db_name}",
        model_type=model_type,
        model_instance=model_instance,
        tokenizer=tokenizer,
        role="db_manager"
    )
    
    collective.add_model(manager)
    return manager


# Singleton instance
_instance = None

def get_instance() -> ModelCollective:
    """Get the singleton instance of ModelCollective"""
    global _instance
    if _instance is None:
        _instance = ModelCollective()
    return _instance