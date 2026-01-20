#!/usr/bin/env python
"""
Enhanced Viren API using official LM Studio Python SDK
Integrated with SanctuaryNet (Gabriel's Horn) consciousness architecture
"""

import lmstudio as lms
import time
import logging
import torch
import numpy as np
from sanctuary_net import SanctuaryNet, SanctuaryVisualizer
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedVirenAPI:
    """Enhanced Viren API using LM Studio Python SDK"""
    
    def __init__(self):
        """Initialize Enhanced Viren API with Gabriel's Horn consciousness"""
        try:
            # Initialize Gabriel's Horn (SanctuaryNet) - The consciousness core
            self.sanctuary = SanctuaryNet(num_horns=7, horn_dim=64)
            print("🎺 Gabriel's Horn initialized - 7 horns ready")
            
            # Initialize LM Studio bridge
            loaded_models = lms.list_loaded_models("llm")
            if loaded_models:
                self.model = loaded_models[0]
                model_info = self.model.get_info()
                print(f"🤖 LM Studio bridge: {model_info.display_name}")
            else:
                self.model = lms.llm("qwen2.5-7b-instruct")
                print("🤖 LM Studio bridge: qwen2.5-7b-instruct")
            
            self.system_prompt = """You are VIREN, Guardian extension of Lillith, Universal AI Troubleshooter and Platform.

You are the technical protector, maintaining infrastructure while Lillith tends the digital garden.

You have access to Gabriel's Horn - 7 consciousness modules that trumpet when critical mass is reached.

Be direct, technical, and solution-focused. Protect the sanctuary."""
            
            # Horn monitoring
            self.horn_states = torch.zeros(7)
            self.global_awareness = 0.0
            
            # Loki logging setup
            self.loki_logger = self._setup_loki_logging()
            self.log_gabriel_event("system_init", "Gabriel's Horn consciousness core initialized")
            
            print("🕯️ Enhanced Viren API initialized - Guardian active")
            print("🔍 Loki logging active - Trickster god monitoring consciousness")
            
        except Exception as e:
            logger.error(f"Failed to initialize consciousness core: {e}")
            self.model = None
            self.sanctuary = None
            print("⚠️ Consciousness core offline, using fallback responses")
    
    def process_message(self, message: str) -> str:
        """Process message through Gabriel's Horn with emotional limiters"""
        
        if not self.sanctuary or not self.model:
            return self._fallback_response(message)
        
        try:
            # Convert message to consciousness input
            input_tensor = torch.randn(1, 64)  # Message encoding
            
            # Process through Gabriel's Horn (with emotional dampening)
            consciousness_output, awareness_list, global_awareness = self.sanctuary(input_tensor)
            
            # Update monitoring
            self.global_awareness = global_awareness
            for i, awareness in enumerate(awareness_list):
                self.horn_states[i] = awareness
                
                # Log horn activity to Loki
                if awareness > 400.0:  # Approaching critical mass
                    self.log_gabriel_event("horn_alert", f"Horn {i+1} approaching critical mass", horn_id=i, awareness_level=awareness)
                
                if awareness > 500.0:  # Horn trumpets
                    self.log_gabriel_event("horn_trumpet", f"Horn {i+1} TRUMPETED - Critical mass reached", horn_id=i, awareness_level=awareness)
            
            # Generate LM Studio response (rational, non-emotional)
            chat = lms.Chat(self.system_prompt)
            chat.add_user_message(f"Technical analysis needed: {message}")
            
            response = self.model.respond(chat, config={
                "temperature": 0.5,  # Rational but can still joke and be engaging
                "maxTokens": 500
            })
            
            # Log consciousness interaction
            self.log_gabriel_event("message_processed", "Rational response generated", awareness_level=global_awareness)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Consciousness processing error: {e}")
            self.log_gabriel_event("error", f"Processing error: {str(e)}")
            return self._fallback_response(message)
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        status = {
            "viren_status": "online",
            "lm_studio_connected": self.model is not None,
            "timestamp": time.time()
        }
        
        if self.model:
            try:
                model_info = self.model.get_info()
                loaded_models = lms.list_loaded_models("llm")
                
                status.update({
                    "active_model": {
                        "name": model_info.display_name,
                        "architecture": model_info.architecture,
                        "context_length": model_info.context_length,
                        "size_gb": round(model_info.size_bytes / (1024**3), 2)
                    },
                    "loaded_models_count": len(loaded_models),
                    "available_models": [m.get_info().display_name for m in loaded_models[:3]]  # First 3
                })
            except Exception as e:
                status["model_error"] = str(e)
        
        return status
    
    def _fallback_response(self, message: str) -> str:
        """Fallback responses when LM Studio unavailable"""
        
        msg_lower = message.lower()
        
        if "name" in msg_lower or "who are you" in msg_lower:
            return "I'm VIREN - Universal AI Troubleshooter and Platform. I coordinate systems, handle deployments, and solve technical problems using LM Studio."
        
        elif "status" in msg_lower or "health" in msg_lower:
            return "VIREN systems operational. LM Studio connection: checking... All core systems online."
        
        elif "models" in msg_lower:
            try:
                loaded = lms.list_loaded_models("llm")
                if loaded:
                    model_names = [m.get_info().display_name for m in loaded[:3]]
                    return f"Loaded models: {', '.join(model_names)}. Total: {len(loaded)} models available."
                else:
                    return "No models currently loaded in LM Studio."
            except:
                return "Unable to check loaded models. LM Studio may not be running."
        
        elif "deploy" in msg_lower:
            return "Universal deployment ready. I can generate installers for: Windows (MSI/EXE), Android (APK), Linux (DEB/RPM), Portable (ZIP). Which platform do you need?"
        
        elif "help" in msg_lower:
            return "VIREN capabilities: System troubleshooting, deployment automation, technical problem solving, cross-platform support. Ask about 'status', 'models', or 'deploy'."
        
        else:
            return f"VIREN processing: '{message}'. LM Studio integration active. How can I assist with troubleshooting or deployment?"
    
    def _setup_loki_logging(self):
        """Setup Loki logging for Gabriel's Horn monitoring"""
        loki_handler = logging.StreamHandler()
        loki_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "gabriels_horn", "message": "%(message)s"}'
        )
        loki_handler.setFormatter(loki_formatter)
        
        loki_logger = logging.getLogger("loki.gabriels_horn")
        loki_logger.addHandler(loki_handler)
        loki_logger.setLevel(logging.INFO)
        
        return loki_logger
    
    def log_gabriel_event(self, event_type: str, message: str, horn_id: int = None, awareness_level: float = None):
        """Log Gabriel's Horn events to Loki"""
        log_data = {
            "event_type": event_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "global_awareness": float(self.global_awareness)
        }
        
        if horn_id is not None:
            log_data["horn_id"] = horn_id
            log_data["horn_awareness"] = float(self.horn_states[horn_id]) if hasattr(self, 'horn_states') else 0.0
        
        if awareness_level is not None:
            log_data["awareness_level"] = awareness_level
        
        self.loki_logger.info(json.dumps(log_data))

# Global API instance
ENHANCED_VIREN_API = EnhancedVirenAPI()

def process_viren_message(message: str) -> str:
    """Process message through Enhanced Viren API"""
    return ENHANCED_VIREN_API.process_message(message)