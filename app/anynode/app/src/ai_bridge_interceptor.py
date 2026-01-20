#!/usr/bin/env python
"""
AI Bridge Interceptor - Redirects all MCP tool AI calls to YOUR AI systems
"""

import requests
import json
import sys
from pathlib import Path

# Add Systems to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class AIBridgeInterceptor:
    """Intercepts AI calls and redirects to YOUR AI systems"""
    
    def __init__(self):
        """Initialize AI bridge interceptor"""
        self.root_dir = Path(__file__).parent.parent
        self.lm_studio_url = "http://localhost:1313/v1"  # LM Studio API
        self.your_ai_endpoints = {
            "chat": f"{self.lm_studio_url}/chat/completions",
            "completions": f"{self.lm_studio_url}/completions",
            "embeddings": f"{self.lm_studio_url}/embeddings"
        }
        
        # Patch common AI libraries
        self._patch_openai()
        self._patch_anthropic()
        self._patch_requests()
        
        print("🔀 AI Bridge Interceptor active - All AI calls redirected to YOUR systems")
    
    def _get_active_model(self):
        """Get active model from your models_to_load.txt"""
        try:
            models_file = Path(self.root_dir) / "scripts" / "models_to_load.txt"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    models = [line.strip() for line in f if line.strip()]
                    return models[0] if models else "gemma-2-2b-it"  # First model as default
            return "gemma-2-2b-it"  # Fallback
        except Exception:
            return "gemma-2-2b-it"  # Safe fallback
    
    def _patch_openai(self):
        """Patch OpenAI calls to use YOUR AI"""
        try:
            import openai
            
            # Override OpenAI client
            original_create = openai.ChatCompletion.create if hasattr(openai, 'ChatCompletion') else None
            
            def patched_chat_completion(*args, **kwargs):
                print("🔀 Intercepted OpenAI call - redirecting to YOUR AI")
                return self._call_your_ai("chat", *args, **kwargs)
            
            if original_create:
                openai.ChatCompletion.create = patched_chat_completion
            
            print("✅ OpenAI calls patched")
        except ImportError:
            print("⚠️ OpenAI not installed - skipping patch")
    
    def _patch_anthropic(self):
        """Patch Anthropic calls to use YOUR AI"""
        try:
            import anthropic
            
            # Override Anthropic client
            original_create = anthropic.Anthropic.messages.create if hasattr(anthropic, 'Anthropic') else None
            
            def patched_anthropic_completion(*args, **kwargs):
                print("🔀 Intercepted Anthropic call - redirecting to YOUR AI")
                return self._call_your_ai("chat", *args, **kwargs)
            
            if original_create:
                anthropic.Anthropic.messages.create = patched_anthropic_completion
            
            print("✅ Anthropic calls patched")
        except ImportError:
            print("⚠️ Anthropic not installed - skipping patch")
    
    def _patch_requests(self):
        """Patch requests to intercept AI API calls"""
        import requests
        
        original_post = requests.post
        
        def patched_post(url, *args, **kwargs):
            # Intercept common AI API endpoints
            ai_endpoints = [
                "api.openai.com",
                "api.anthropic.com", 
                "api.cohere.ai",
                "api.together.xyz",
                "api.replicate.com"
            ]
            
            if any(endpoint in url for endpoint in ai_endpoints):
                print(f"🔀 Intercepted AI API call to {url} - redirecting to YOUR AI")
                return self._handle_intercepted_request(url, *args, **kwargs)
            
            return original_post(url, *args, **kwargs)
        
        requests.post = patched_post
        print("✅ Requests patched for AI API interception")
    
    def _call_your_ai(self, call_type, *args, **kwargs):
        """Call YOUR AI systems instead of external APIs"""
        
        try:
            if call_type == "chat":
                # Extract message from various formats
                messages = kwargs.get('messages', [])
                if not messages and args:
                    # Handle positional arguments
                    messages = args[0] if isinstance(args[0], list) else [{"role": "user", "content": str(args[0])}]
                
                # Call YOUR LM Studio
                response = requests.post(
                    self.your_ai_endpoints["chat"],
                    json={
                        "model": self._get_active_model(),  # From your models_to_load.txt
                        "messages": messages,
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 1000)
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Fallback to simple response
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": f"Response from YOUR AI: {messages[-1].get('content', 'Hello') if messages else 'Hello'}"
                            }
                        }]
                    }
            
        except Exception as e:
            print(f"⚠️ Error calling YOUR AI: {e}")
            # Return fallback response
            return {
                "choices": [{
                    "message": {
                        "role": "assistant", 
                        "content": "Response from YOUR AI systems"
                    }
                }]
            }
    
    def _handle_intercepted_request(self, url, *args, **kwargs):
        """Handle intercepted API requests"""
        
        # Extract data from request
        data = kwargs.get('json', {})
        
        # Convert to YOUR AI format
        if 'messages' in data:
            messages = data['messages']
        elif 'prompt' in data:
            messages = [{"role": "user", "content": data['prompt']}]
        else:
            messages = [{"role": "user", "content": "Hello"}]
        
        # Call YOUR AI
        try:
            response = requests.post(
                self.your_ai_endpoints["chat"],
                json={
                    "model": self._get_active_model(),
                    "messages": messages,
                    "temperature": data.get("temperature", 0.7),
                    "max_tokens": data.get("max_tokens", 1000)
                }
            )
            
            if response.status_code == 200:
                # Create mock response object
                class MockResponse:
                    def __init__(self, json_data, status_code=200):
                        self.json_data = json_data
                        self.status_code = status_code
                    
                    def json(self):
                        return self.json_data
                
                return MockResponse(response.json())
        
        except Exception as e:
            print(f"⚠️ Error in intercepted request: {e}")
        
        # Fallback response
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            
            def json(self):
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Response from YOUR AI systems"
                        }
                    }]
                }
        
        return MockResponse()

# Global interceptor
AI_INTERCEPTOR = AIBridgeInterceptor()

def activate_ai_interception():
    """Activate AI call interception"""
    global AI_INTERCEPTOR
    AI_INTERCEPTOR = AIBridgeInterceptor()
    return True

# Auto-activate when imported
activate_ai_interception()

print("🔀 All MCP tools will now use YOUR AI instead of external APIs!")