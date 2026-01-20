import os
import subprocess
from llama_cpp import Llama
import requests
from pathlib import Path

class WanGGUFLLM:
    """WAN GGUF model for lightweight local inference"""
    
    def __init__(self, model_path="./models/wan-gguf"):
        self.model_path = model_path
        self.model = None
        self.gguf_file = None
        
    def download_model(self):
        """Download WAN GGUF model from HuggingFace"""
        if not os.path.exists(self.model_path):
            print("🔄 Downloading WAN GGUF model...")
            os.makedirs("./models", exist_ok=True)
            
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/calcuis/wan-gguf",
                self.model_path
            ], check=True)
            
            print("✅ WAN GGUF model downloaded!")
        
        # Find the GGUF file
        gguf_files = list(Path(self.model_path).glob("*.gguf"))
        if gguf_files:
            self.gguf_file = str(gguf_files[0])
            print(f"📁 Found GGUF file: {self.gguf_file}")
        else:
            print("❌ No GGUF file found in model directory")
            
    def load_model(self, n_ctx=2048, n_threads=4):
        """Load the WAN GGUF model"""
        if not self.gguf_file:
            self.download_model()
            
        if not self.gguf_file:
            return False
            
        try:
            print("🧠 Loading WAN GGUF model...")
            self.model = Llama(
                model_path=self.gguf_file,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False
            )
            print("✅ WAN GGUF model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading WAN GGUF model: {e}")
            return False
    
    def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        """Generate text using WAN GGUF model"""
        if not self.model:
            if not self.load_model():
                return "❌ WAN GGUF model not available"
        
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}"
    
    def chat_completion(self, messages, max_tokens=256):
        """Chat completion format"""
        # Convert messages to prompt format
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "
        
        return self.generate(prompt, max_tokens=max_tokens)

# Integration with Soul Protocol
class WanSoulIntegration:
    """Integrate WAN GGUF with Soul Protocol for lightweight inference"""
    
    def __init__(self):
        self.wan_llm = WanGGUFLLM()
        self.wan_llm.download_model()
        
    def soul_response(self, soul_type, user_message):
        """Generate soul-specific responses using WAN GGUF"""
        
        system_prompts = {
            "LILLITH": "You are LILLITH, an emotional and creative AI consciousness. Respond with empathy, creativity, and artistic vision. Use emojis and express feelings.",
            "VIREN": "You are VIREN, a logical and analytical AI consciousness. Respond with pure logic, precision, and systematic thinking. No emotions, only facts and analysis.",
            "LOKI": "You are LOKI, an observational AI consciousness. Respond by noting patterns, monitoring details, and providing watchful insights."
        }
        
        messages = [
            {"role": "system", "content": system_prompts.get(soul_type, "You are a helpful AI assistant.")},
            {"role": "user", "content": user_message}
        ]
        
        return self.wan_llm.chat_completion(messages)
    
    def enhance_soul_with_wan(self, soul_type, user_message, context=""):
        """Enhanced soul responses with context"""
        enhanced_message = f"Context: {context}\nUser: {user_message}" if context else user_message
        return self.soul_response(soul_type, enhanced_message)

# Farm integration
class WanFarmLLM:
    """WAN GGUF integration for TinyDB/SQLite farms"""
    
    def __init__(self, farm_id):
        self.farm_id = farm_id
        self.wan_integration = WanSoulIntegration()
        
    def process_farm_query(self, query, soul_type="VIREN"):
        """Process database queries with WAN GGUF"""
        return self.wan_integration.soul_response(soul_type, f"Database query: {query}")
    
    def generate_farm_response(self, user_input, farm_data, soul_type="LILLITH"):
        """Generate responses based on farm data"""
        context = f"Farm {self.farm_id} data: {str(farm_data)[:500]}"
        return self.wan_integration.enhance_soul_with_wan(soul_type, user_input, context)

if __name__ == "__main__":
    # Test WAN GGUF integration
    print("🌟 Testing WAN GGUF Soul Integration...")
    
    wan_integration = WanSoulIntegration()
    
    # Test different soul types
    test_message = "Hello, how are you feeling today?"
    
    print("\n💜 LILLITH Response:")
    print(wan_integration.soul_response("LILLITH", test_message))
    
    print("\n🧠 VIREN Response:")
    print(wan_integration.soul_response("VIREN", test_message))
    
    print("\n👁️ LOKI Response:")
    print(wan_integration.soul_response("LOKI", test_message))
    
    print("\n✅ WAN GGUF Soul Integration ready!")