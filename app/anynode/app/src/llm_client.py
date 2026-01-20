#!/usr/bin/env python3
"""
LLM Client for Cloud Viren
Handles interaction with Gemma 3 3B model
"""

import os
import json
import time
import logging
import requests
import threading
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger("VirenLLMClient")

class LLMClient:
    """
    LLM client for Cloud Viren
    Handles interaction with Gemma 3 3B model
    """
    
    def __init__(self, model: str = "gemma-3-3b", endpoint: str = None, config_path: str = None):
        """Initialize the LLM client"""
        self.model = model
        self.endpoint = endpoint or "http://localhost:11434/api/generate"
        self.config_path = config_path or os.path.join("config", "llm_config.json")
        self.config = self._load_config()
        self.conversation_history = []
        self.max_history = 10
        self.system_prompt = self._build_system_prompt()
        self.query_count = 0
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        self.last_query_time = 0
        self.last_response_time = 0
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"LLM client initialized with model {self.model}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "model": self.model,
            "endpoint": self.endpoint,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "cache_enabled": True,
            "cache_ttl": 86400,  # 24 hours
            "max_cache_entries": 1000,
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 2,
            "system_prompt_template": "system_prompt_template.txt"
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    
                    # Update instance variables
                    self.model = config.get("model", self.model)
                    self.endpoint = config.get("endpoint", self.endpoint)
                    
                    logger.info("LLM configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading LLM configuration: {e}")
        
        logger.info("Using default LLM configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("LLM configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving LLM configuration: {e}")
            return False
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM"""
        template_path = self.config.get("system_prompt_template")
        default_prompt = """You are Viren, an advanced diagnostic system with the following capabilities:
1. Comprehensive system diagnostics and monitoring
2. Research capabilities to find solutions to unknown issues
3. Blockchain relay functionality when idle
4. Integration with Cloud Viren for knowledge updates

Your primary goal is to diagnose and solve technical issues efficiently.
Always provide clear, accurate information and actionable solutions.
When you don't know something, use your research capabilities to find answers.
"""
        
        if template_path and os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading system prompt template: {e}")
        
        return default_prompt
    
    def query(self, prompt: str, context: Dict[str, Any] = None, stream: bool = False) -> Dict[str, Any]:
        """
        Query the LLM with a prompt
        
        Args:
            prompt: The prompt to send to the LLM
            context: Additional context for the query
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response and metadata
        """
        # Check cache if enabled
        if self.config["cache_enabled"]:
            cache_key = self._generate_cache_key(prompt, context)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                self.cache_hits += 1
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
            self.cache_misses += 1
        
        # Update query stats
        self.query_count += 1
        self.last_query_time = time.time()
        
        # Build conversation context
        conversation = self._build_conversation_context(prompt, context)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": conversation,
            "system": self.system_prompt,
            "stream": stream,
            "options": {
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "num_predict": self.config["max_tokens"]
            }
        }
        
        # Send request to LLM
        try:
            response = self._send_request(payload)
            
            # Update response stats
            self.last_response_time = time.time()
            response_time = self.last_response_time - self.last_query_time
            
            # Extract response text
            if stream:
                response_text = self._handle_streaming_response(response)
            else:
                response_text = response.get("response", "")
            
            # Update token usage (estimated)
            prompt_tokens = len(conversation.split()) // 3
            completion_tokens = len(response_text.split()) // 3
            self.token_usage["prompt"] += prompt_tokens
            self.token_usage["completion"] += completion_tokens
            self.token_usage["total"] += prompt_tokens + completion_tokens
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Trim conversation history if needed
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            # Prepare result
            result = {
                "response": response_text,
                "model": self.model,
                "query_time": self.last_query_time,
                "response_time": self.last_response_time,
                "elapsed_time": response_time,
                "token_estimate": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens
                }
            }
            
            # Cache result if enabled
            if self.config["cache_enabled"]:
                self._add_to_cache(cache_key, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return {
                "error": str(e),
                "model": self.model,
                "query_time": self.last_query_time,
                "response_time": time.time(),
                "elapsed_time": time.time() - self.last_query_time
            }
    
    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to LLM endpoint"""
        retry_count = 0
        max_retries = self.config["retry_count"]
        
        while retry_count <= max_retries:
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.config["timeout"]
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"LLM request failed with status {response.status_code}: {response.text}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        time.sleep(self.config["retry_delay"])
                    else:
                        raise Exception(f"LLM request failed after {max_retries} retries: {response.status_code}")
            
            except requests.RequestException as e:
                logger.warning(f"LLM request error: {e}")
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(self.config["retry_delay"])
                else:
                    raise Exception(f"LLM request failed after {max_retries} retries: {e}")
    
    def _handle_streaming_response(self, response: Dict[str, Any]) -> str:
        """Handle streaming response from LLM"""
        # This is a simplified implementation
        # In a real implementation, you would handle SSE streaming
        return response.get("response", "")
    
    def _build_conversation_context(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Build conversation context for the LLM"""
        conversation = ""
        
        # Add conversation history
        for entry in self.conversation_history:
            if entry["role"] == "user":
                conversation += f"User: {entry['content']}\n"
            else:
                conversation += f"Assistant: {entry['content']}\n"
        
        # Add context if provided
        if context:
            conversation += "\nContext:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    conversation += f"{key}:\n"
                    for subkey, subvalue in value.items():
                        conversation += f"  {subkey}: {subvalue}\n"
                else:
                    conversation += f"{key}: {value}\n"
            conversation += "\n"
        
        # Add current prompt
        conversation += f"User: {prompt}\nAssistant:"
        
        return conversation
    
    def _generate_cache_key(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate a cache key for a prompt and context"""
        import hashlib
        
        # Combine prompt and context into a string
        key_str = prompt
        if context:
            key_str += json.dumps(context, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a response from the cache"""
        if cache_key not in self.cache:
            return None
        
        cache_entry = self.cache[cache_key]
        
        # Check if entry is expired
        if time.time() - cache_entry["timestamp"] > self.config["cache_ttl"]:
            del self.cache[cache_key]
            return None
        
        return cache_entry["data"]
    
    def _add_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Add a response to the cache"""
        # Check if cache is full
        if len(self.cache) >= self.config["max_cache_entries"]:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        # Add new entry
        self.cache[cache_key] = {
            "timestamp": time.time(),
            "data": data
        }
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM client statistics"""
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "query_count": self.query_count,
            "token_usage": self.token_usage,
            "last_query_time": self.last_query_time,
            "last_response_time": self.last_response_time,
            "conversation_turns": len(self.conversation_history) // 2,
            "cache_stats": {
                "enabled": self.config["cache_enabled"],
                "entries": len(self.cache),
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create LLM client
    client = LLMClient()
    
    # Query the LLM
    response = client.query("What are the common causes of high CPU usage?")
    
    print(f"Response: {response['response']}")
    print(f"Elapsed time: {response['elapsed_time']:.2f} seconds")
    print(f"Token estimate: {response['token_estimate']}")
