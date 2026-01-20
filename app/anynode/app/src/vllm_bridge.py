# vllm_bridge.py
# Purpose: Bridge module to connect Viren with vLLM for high-performance inference
# Location: /root/bridge/vllm_bridge.py

import requests
import logging
import json
import os
from typing import Dict, Any, Optional, List, Union

# Configure logging
logger = logging.getLogger("vllm_bridge")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/boot_logs/vllm_bridge.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Default configuration
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

def is_available() -> bool:
    """Check if vLLM is available and running."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"vLLM not available: {e}")
        return False

def list_models() -> List[str]:
    """Get list of available models in vLLM."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        response.raise_for_status()
        return [model["id"] for model in response.json().get("data", [])]
    except Exception as e:
        logger.error(f"Failed to list vLLM models: {e}")
        return []

def query(prompt: str, model: Optional[str] = None, 
          max_tokens: int = 256, temperature: float = 0.7, 
          stream: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Query vLLM with a prompt.
    
    Args:
        prompt: The input prompt
        model: Optional model name (if None, uses default model)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        stream: Whether to stream the response
        
    Returns:
        Generated text or streaming response object
    """
    # Determine if this is a chat model or completion model based on model name
    is_chat_model = True
    if model:
        # Simple heuristic: models without "chat", "instruct", etc. are likely completion models
        non_chat_indicators = ["base", "code-", "coder", "starcoder"]
        if any(indicator in model.lower() for indicator in non_chat_indicators) and not any(
            chat_indicator in model.lower() for chat_indicator in ["chat", "instruct", "conv"]
        ):
            is_chat_model = False
    
    if is_chat_model:
        endpoint = "v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if model:
            payload["model"] = model
    else:
        endpoint = "v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if model:
            payload["model"] = model
    
    url = f"{VLLM_BASE_URL}/{endpoint}"
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
            
            # Extract the generated text based on API response format
            if "chat" in endpoint:
                return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                return result.get("choices", [{}])[0].get("text", "").strip()
                
    except Exception as e:
        error_msg = f"vLLM query failed: {e}"
        logger.error(error_msg)
        return error_msg

def stream_process(response) -> str:
    """Process a streaming response from vLLM."""
    full_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]  # Remove 'data: ' prefix
                if line.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            full_text += content
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue
    print()  # New line after streaming completes
    return full_text

def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        response = requests.get(f"{VLLM_BASE_URL}/v1/models/{model_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        return {}

def load_model(model_id: str, model_path: Optional[str] = None) -> bool:
    """
    Load a model into vLLM.
    
    Args:
        model_id: The ID to assign to the model
        model_path: Optional path to the model files (if different from model_id)
        
    Returns:
        True if successful, False otherwise
    """
    url = f"{VLLM_BASE_URL}/v1/models/load"
    payload = {
        "model_id": model_id
    }
    if model_path:
        payload["model_path"] = model_path
        
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully loaded model {model_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        return False

if __name__ == "__main__":
    # Simple test
    if is_available():
        print("vLLM is available. Testing with a simple query...")
        response = query("What is the capital of France?")
        print(f"Response: {response}")
    else:
        print("vLLM is not available. Please ensure the service is running.")
