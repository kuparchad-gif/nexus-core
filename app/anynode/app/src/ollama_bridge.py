# ollama_bridge.py
# Purpose: Bridge module to connect Viren with Ollama for local inference
# Location: /root/bridge/ollama_bridge.py

import requests
import logging
import json
import os
from typing import Dict, Any, Optional, List, Union

# Configure logging
logger = logging.getLogger("ollama_bridge")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/boot_logs/ollama_bridge.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Default configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

def is_available() -> bool:
    """Check if Ollama is available and running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return False

def list_models() -> List[str]:
    """Get list of available models in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return []

def query(prompt: str, model: str = "gemma:2b", 
          max_tokens: int = 256, temperature: float = 0.7, 
          stream: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Query Ollama with a prompt.
    
    Args:
        prompt: The input prompt
        model: Model name (defaults to gemma:2b)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        stream: Whether to stream the response
        
    Returns:
        Generated text or streaming response object
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        if stream:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            return stream_process(response)
        else:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
                
    except Exception as e:
        error_msg = f"Ollama query failed: {e}"
        logger.error(error_msg)
        return error_msg

def stream_process(response) -> str:
    """Process a streaming response from Ollama."""
    full_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    content = chunk["response"]
                    full_text += content
                    print(content, end="", flush=True)
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
    print()  # New line after streaming completes
    return full_text

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/show", 
            json={"name": model_name},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        return {}

def pull_model(model_name: str) -> bool:
    """
    Pull a model from Ollama library.
    
    Args:
        model_name: The name of the model to pull (e.g., "gemma:2b")
        
    Returns:
        True if successful, False otherwise
    """
    url = f"{OLLAMA_BASE_URL}/api/pull"
    payload = {"name": model_name}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully pulled model {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        return False

def create_model(model_name: str, model_path: str, template: Optional[str] = None) -> bool:
    """
    Create a model in Ollama from local files.
    
    Args:
        model_name: The name to assign to the model
        model_path: Path to the model files
        template: Optional template for the model
        
    Returns:
        True if successful, False otherwise
    """
    url = f"{OLLAMA_BASE_URL}/api/create"
    
    # Build Modelfile content
    modelfile = f"FROM {model_path}\n"
    if template:
        modelfile += f"TEMPLATE {template}\n"
    
    payload = {
        "name": model_name,
        "modelfile": modelfile
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully created model {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        return False

def map_model_to_ollama(model_id: str) -> str:
    """
    Map a model ID from the manifest to an Ollama model name.
    
    Args:
        model_id: The model ID from the manifest
        
    Returns:
        Ollama model name
    """
    # Map common model names to Ollama equivalents
    mapping = {
        "gemma-3-1b-it-qat": "gemma:1b",
        "gemma-3-4b-it": "gemma:4b",
        "gemma-3-7b-it": "gemma:7b",
        "llama-3.1-8b-instruct": "llama3:8b",
        "mistral-7b-instruct-v0.3": "mistral:7b",
        "phi-3.1-mini-128k-instruct": "phi3:mini",
        "qwen2.5-7b-instruct": "qwen2:7b",
        "qwen3-1.7b": "qwen3:1.7b"
    }
    
    # Try direct mapping
    if model_id in mapping:
        return mapping[model_id]
    
    # Try to infer from model ID
    if "gemma" in model_id.lower():
        size = "2b"
        if "1b" in model_id.lower():
            size = "1b"
        elif "4b" in model_id.lower():
            size = "4b"
        elif "7b" in model_id.lower():
            size = "7b"
        return f"gemma:{size}"
    
    if "llama" in model_id.lower():
        return "llama3"
    
    if "mistral" in model_id.lower():
        return "mistral"
    
    if "phi" in model_id.lower():
        return "phi3:mini"
    
    # Default fallback
    return "gemma:2b"

if __name__ == "__main__":
    # Simple test
    if is_available():
        print("Ollama is available. Testing with a simple query...")
        response = query("What is the capital of France?")
        print(f"Response: {response}")
    else:
        print("Ollama is not available. Please ensure the service is running.")
