#!/usr/bin/env python3
"""
Moshi Integration for Viren Cloud
Integrates the Moshi speech-text foundation model as Viren's voice interface
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MoshiIntegration")

class MoshiIntegration:
    """Integrates Moshi model with Viren Cloud"""
    
    def __init__(self, config_path: str = None):
        """Initialize the Moshi integration"""
        self.config_path = config_path or os.path.join('C:/Viren/config', 'moshi_config.json')
        self.model_path = os.environ.get("MOSHI_MODEL_PATH", "C:/Engineers/root/models/moshiko-pytorch-bf16")
        self.config = {}
        self.moshi_model = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load Moshi configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default configuration
                self.config = {
                    "model_path": self.model_path,
                    "streaming": {
                        "enabled": True,
                        "buffer_size": 4096,
                        "sample_rate": 24000,
                        "channels": 1
                    },
                    "websocket": {
                        "port": 8765,
                        "host": "0.0.0.0"
                    },
                    "voice": {
                        "speed": 1.0,
                        "volume": 1.0
                    }
                }
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading Moshi configuration: {e}")
            self.config = {}
    
    def _save_config(self) -> None:
        """Save Moshi configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Moshi configuration: {e}")
    
    def initialize(self) -> bool:
        """Initialize the Moshi integration"""
        logger.info("Initializing Moshi Integration")
        
        try:
            # Check if model files exist
            model_path = self.config.get("model_path", self.model_path)
            model_file = os.path.join(model_path, "model.safetensors")
            tokenizer_file = os.path.join(model_path, "tokenizer_spm_32k_3.model")
            
            if not os.path.exists(model_file) or not os.path.exists(tokenizer_file):
                logger.error(f"Moshi model files not found at {model_path}")
                return False
            
            # Import Moshi library
            try:
                # First, add moshi to Python path if needed
                moshi_lib_path = os.environ.get("MOSHI_LIB_PATH", "")
                if moshi_lib_path and moshi_lib_path not in sys.path:
                    sys.path.append(moshi_lib_path)
                
                import moshi
                logger.info("Successfully imported Moshi library")
            except ImportError:
                logger.error("Failed to import Moshi library. Make sure it's installed or MOSHI_LIB_PATH is set correctly")
                return False
            
            # Load Moshi model
            try:
                self.moshi_model = moshi.MoshiModel(
                    model_path=model_file,
                    tokenizer_path=tokenizer_file,
                    device="cuda" if moshi.is_cuda_available() else "cpu"
                )
                logger.info("Successfully loaded Moshi model")
                return True
            except Exception as e:
                logger.error(f"Failed to load Moshi model: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Moshi integration: {e}")
            return False
    
    def start_websocket_server(self) -> bool:
        """Start the Moshi websocket server"""
        if not self.moshi_model:
            logger.error("Moshi model not initialized")
            return False
        
        try:
            import asyncio
            import websockets
            from moshi.server import MoshiServer
            
            websocket_config = self.config.get("websocket", {})
            host = websocket_config.get("host", "0.0.0.0")
            port = websocket_config.get("port", 8765)
            
            # Create and start server
            server = MoshiServer(self.moshi_model)
            
            # Run the server
            asyncio.run(server.run(host, port))
            
            logger.info(f"Moshi websocket server started on {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Error starting Moshi websocket server: {e}")
            return False
    
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio data and get response"""
        if not self.moshi_model:
            return {"error": "Moshi model not initialized"}
        
        try:
            # Process audio with Moshi
            response = self.moshi_model.process_audio(audio_data)
            
            return {
                "success": True,
                "response_audio": response.get("audio"),
                "response_text": response.get("text"),
                "latency_ms": response.get("latency_ms")
            }
        except Exception as e:
            logger.error(f"Error processing audio with Moshi: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get Moshi integration status"""
        return {
            "initialized": self.moshi_model is not None,
            "model_path": self.config.get("model_path"),
            "streaming_enabled": self.config.get("streaming", {}).get("enabled", True),
            "device": "cuda" if hasattr(self, "moshi_model") and self.moshi_model and hasattr(self.moshi_model, "device") and "cuda" in self.moshi_model.device else "cpu"
        }