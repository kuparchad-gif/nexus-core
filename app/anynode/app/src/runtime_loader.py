# runtime_loader.py
# Purpose: Select the optimal model execution backend (MLX, GGUF, Transformers, vLLM, etc.)
# Location: /root/services/runtime_loader.py

import platform
import importlib
import logging
import os
import json
from typing import Dict, Any, Optional, List, Union

ACTIVE_BACKEND = None

# Configure runtime logger
logger = logging.getLogger("runtime_loader")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/boot_logs/runtime.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def detect_backend():
    """
    Detect the best available backend based on environment and available libraries.
    Priority order:
    1. vLLM (if available and GPU present)
    2. MLX (on macOS with Apple Silicon)
    3. Ollama (if running)
    4. LM Studio (on Windows if running)
    5. Transformers (fallback)
    """
    system = platform.system().lower()
    backend = None

    # Check for vLLM first (highest priority if GPU available)
    try:
        import bridge.vllm_bridge as vllm
        if vllm.is_available():
            logger.info("vLLM backend detected and available")
            return "vllm"
    except ImportError:
        logger.debug("vLLM not available")

    # Check for platform-specific optimized backends
    try:
        if system == "darwin":
            # Check for Apple Silicon
            if platform.processor() == "arm":
                import bridge.mlx_bridge as mlx_bridge
                backend = "mlx"
                logger.info("MLX backend selected for Apple Silicon")
            else:
                # For Intel Macs, try Ollama
                import bridge.ollama_bridge as ollama
                if ollama.is_available():
                    backend = "ollama"
                    logger.info("Ollama backend selected for macOS Intel")
        elif system == "windows":
            # Try LM Studio first on Windows
            import bridge.lmstudio_bridge as lmstudio
            backend = "lmstudio"
            logger.info("LM Studio backend selected for Windows")
        else:
            # For Linux, try Ollama first
            try:
                import bridge.ollama_bridge as ollama
                if ollama.is_available():
                    backend = "ollama"
                    logger.info("Ollama backend selected for Linux")
            except ImportError:
                # Fallback to transformers
                import transformers
                backend = "transformers"
                logger.info("Transformers backend selected as fallback")
    except ImportError as e:
        logger.warning(f"Backend import failed: {e}")
        # Final fallback to transformers
        try:
            import transformers
            backend = "transformers"
            logger.info("Transformers backend selected as fallback after import error")
        except ImportError:
            logger.error("No valid backend found")
            backend = "none"

    logger.info(f"Selected backend: {backend}")
    return backend


def run_model(prompt: str, model_name: str = None, role: str = None, subrole: str = None, max_tokens: int = 256):
    """
    Run a model with the given prompt.
    
    Args:
        prompt: The input prompt
        model_name: Specific model to use (overrides role-based selection)
        role: Role to use for model selection
        subrole: Subrole to use for model selection
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text
    """
    # Try to use the model router if available
    try:
        return run_model_with_router(prompt, model_name, role, subrole, max_tokens)
    except ImportError:
        # Fall back to direct backend usage
        pass
    
    global ACTIVE_BACKEND
    if ACTIVE_BACKEND is None:
        ACTIVE_BACKEND = detect_backend()
    
    # Determine which model to use
    if not model_name and (role or subrole):
        from config.model_config import get_model_for_role
        model_name = get_model_for_role(role, subrole)
    
    # If still no model specified, use bootstrap model
    if not model_name:
        from config.model_config import get_bootstrap_model
        model_name = get_bootstrap_model()
    
    logger.info(f"Using model {model_name} on backend {ACTIVE_BACKEND}")
    
    if ACTIVE_BACKEND == "vllm":
        import bridge.vllm_bridge as vllm
        return vllm.query(prompt, model=model_name, max_tokens=max_tokens)

    elif ACTIVE_BACKEND == "mlx":
        import bridge.mlx_bridge as mlx
        return mlx.query(prompt, model=model_name)

    elif ACTIVE_BACKEND == "lmstudio":
        import bridge.lmstudio_bridge as lmstudio
        return lmstudio.query(prompt, model_name)

    elif ACTIVE_BACKEND == "ollama":
        import bridge.ollama_bridge as ollama
        # Map model name to Ollama format
        ollama_model = ollama.map_model_to_ollama(model_name)
        return ollama.query(prompt, model=ollama_model, max_tokens=max_tokens)

    elif ACTIVE_BACKEND == "transformers":
        from transformers import pipeline
        model = model_name if model_name else "google/gemma-2b"
        pipe = pipeline("text-generation", model=model)
        return pipe(prompt, max_length=max_tokens)[0]["generated_text"]

    else:
        raise RuntimeError("No valid model backend available.")


def run_model_with_router(prompt: str, model_name: str = None, role: str = None, subrole: str = None, max_tokens: int = 256):
    """
    Run a model using the model router.
    
    Args:
        prompt: The input prompt
        model_name: Specific model to use (overrides role-based selection)
        role: Role to use for model selection
        subrole: Subrole to use for model selection
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text
    """
    # Determine which model to use
    if not model_name and (role or subrole):
        from config.model_config import get_model_for_role
        model_name = get_model_for_role(role, subrole)
    
    # If still no model specified, use bootstrap model
    if not model_name:
        from config.model_config import get_bootstrap_model
        model_name = get_bootstrap_model()
    
    # Use the model router
    from bridge.model_router import query
    return query(prompt, model_name, max_tokens=max_tokens)


def send_message(from_module: str, to_module: str, message: str, **kwargs):
    """
    Send a message from one module to another.
    
    Args:
        from_module: The sending module (e.g., "consciousness")
        to_module: The receiving module (e.g., "memory")
        message: The message to send
        **kwargs: Additional arguments for the query
        
    Returns:
        Response from the receiving module
    """
    try:
        from bridge.model_router import send_message as router_send_message
        return router_send_message(from_module, to_module, message, **kwargs)
    except ImportError:
        # Fall back to direct model usage
        from config.model_config import get_model_for_role
        to_model = get_model_for_role(to_module)
        formatted_message = f"[FROM:{from_module}] {message}"
        return run_model(formatted_message, model_name=to_model, **kwargs)


if __name__ == "__main__":
    # Test with bootstrap model
    from config.model_config import get_bootstrap_model, get_model_for_role
    
    bootstrap_model = get_bootstrap_model()
    print(f"Bootstrap model: {bootstrap_model}")
    
    consciousness_model = get_model_for_role("consciousness")
    print(f"Consciousness model: {consciousness_model}")
    
    out = run_model("Hello, who are you?", role="bootstrap")
    print("Model Output:", out)
