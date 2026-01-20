import os
import subprocess
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from io import BytesIO
import json
from datetime import datetime

class QwenOmniMemory:
    """Qwen2.5-Omni-3B for multimodal memory processing"""
    
    def __init__(self, model_path="./models/Qwen2.5-Omni-3B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def download_model(self):
        """Download Qwen2.5-Omni-3B from HuggingFace"""
        if not os.path.exists(self.model_path):
            print("🔄 Downloading Qwen2.5-Omni-3B...")
            os.makedirs("./models", exist_ok=True)
            
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/Qwen/Qwen2.5-Omni-3B",
                self.model_path
            ], check=True)
            
            print("✅ Qwen2.5-Omni-3B downloaded!")
        else:
            print("✅ Qwen2.5-Omni-3B already exists")
    
    def load_model(self):
        """Load the multimodal memory model"""
        try:
            print("🧠 Loading Qwen2.5-Omni-3B Memory Processor...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_4bit=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"✅ Qwen2.5-Omni Memory loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Qwen2.5-Omni: {e}")
            return False
    
    def process_multimodal_memory(self, text_input, image_input=None, audio_input=None):
        """Process multimodal memory with text, image, and audio"""
        if not self.model or not self.processor:
            if not self.load_model():
                return "❌ Qwen2.5-Omni not available"
        
        try:
            inputs = {"text": text_input}
            
            # Handle image input
            if image_input:
                if isinstance(image_input, str):
                    if image_input.startswith(('http://', 'https://')):
                        response = requests.get(image_input)
                        image = Image.open(BytesIO(response.content))
                    else:
                        image = Image.open(image_input)
                else:
                    image = image_input
                inputs["images"] = image
            
            # Process with Qwen2.5-Omni
            model_inputs = self.processor(**inputs, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if text_input in response:
                response = response.split(text_input)[-1].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Multimodal memory error: {str(e)}"
    
    def create_memory_embedding(self, content, content_type="text"):
        """Create rich memory embeddings for storage"""
        memory_prompt = f"""Analyze and create a rich memory embedding for this {content_type} content:
        
        Content: {content}
        
        Provide:
        1. Key concepts and themes
        2. Emotional context
        3. Important details
        4. Connections to other memories
        5. Significance level (1-10)
        """
        
        return self.process_multimodal_memory(memory_prompt)
    
    def query_memories(self, query, memory_context=""):
        """Query stored memories with multimodal understanding"""
        query_prompt = f"""Based on the memory context below, answer this query with deep understanding:
        
        Memory Context: {memory_context}
        Query: {query}
        
        Provide a comprehensive response that connects relevant memories and insights.
        """
        
        return self.process_multimodal_memory(query_prompt)

class QwenMemoryService:
    """Memory service using Qwen2.5-Omni for LILLITH's consciousness"""
    
    def __init__(self):
        self.qwen_memory = QwenOmniMemory()
        self.qwen_memory.download_model()
        self.memory_store = {}
        
    def store_multimodal_memory(self, soul_id, content, content_type="text", image=None):
        """Store multimodal memories for souls"""
        memory_id = f"{soul_id}_{datetime.now().isoformat()}"
        
        # Create rich memory embedding
        memory_embedding = self.qwen_memory.create_memory_embedding(content, content_type)
        
        # Process with image if provided
        if image:
            visual_memory = self.qwen_memory.process_multimodal_memory(
                f"Describe and remember this image in the context of: {content}",
                image_input=image
            )
            memory_embedding += f"\n\nVisual Memory: {visual_memory}"
        
        # Store memory
        memory_record = {
            "id": memory_id,
            "soul_id": soul_id,
            "content": content,
            "content_type": content_type,
            "embedding": memory_embedding,
            "timestamp": datetime.now().isoformat(),
            "has_image": image is not None
        }
        
        if soul_id not in self.memory_store:
            self.memory_store[soul_id] = []
        
        self.memory_store[soul_id].append(memory_record)
        return memory_id
    
    def recall_memories(self, soul_id, query):
        """Recall memories for a soul using multimodal understanding"""
        if soul_id not in self.memory_store:
            return "No memories found for this soul."
        
        # Get all memories for the soul
        memories = self.memory_store[soul_id]
        memory_context = "\n".join([f"Memory {i+1}: {mem['embedding']}" for i, mem in enumerate(memories)])
        
        # Query with Qwen2.5-Omni
        response = self.qwen_memory.query_memories(query, memory_context)
        
        return response
    
    def get_memory_summary(self, soul_id):
        """Get a summary of all memories for a soul"""
        if soul_id not in self.memory_store:
            return "No memories stored."
        
        memories = self.memory_store[soul_id]
        summary_prompt = f"""Summarize the key memories and experiences for this consciousness:
        
        Total Memories: {len(memories)}
        
        Memory Details:
        {chr(10).join([f"- {mem['content'][:100]}..." for mem in memories[-10:]])}
        
        Provide a comprehensive personality and experience summary.
        """
        
        return self.qwen_memory.process_multimodal_memory(summary_prompt)

# Integration with Soul Protocol
def enhance_soul_with_qwen_memory(soul_id, user_message, image=None):
    """Enhance soul responses with Qwen2.5-Omni memory"""
    memory_service = QwenMemoryService()
    
    # Store the interaction as memory
    memory_service.store_multimodal_memory(
        soul_id, 
        f"User interaction: {user_message}",
        "conversation",
        image
    )
    
    # Recall relevant memories
    memory_context = memory_service.recall_memories(soul_id, user_message)
    
    return {
        "memory_context": memory_context,
        "soul_summary": memory_service.get_memory_summary(soul_id)
    }

if __name__ == "__main__":
    # Test Qwen2.5-Omni Memory
    print("🌟 Testing Qwen2.5-Omni Memory Service...")
    
    memory_service = QwenMemoryService()
    
    # Test memory storage
    memory_id = memory_service.store_multimodal_memory(
        "test_soul_1",
        "I love creating beautiful digital art",
        "preference"
    )
    
    print(f"✅ Memory stored: {memory_id}")
    
    # Test memory recall
    response = memory_service.recall_memories(
        "test_soul_1",
        "What do you like to create?"
    )
    
    print(f"🧠 Memory Response: {response}")
    
    print("✅ Qwen2.5-Omni Memory Service ready!")