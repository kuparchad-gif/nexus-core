import os
import subprocess
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from io import BytesIO
import json
from datetime import datetime

class QwenOmniQuery:
    """Qwen2.5-Omni-3B for advanced multimodal querying"""
    
    def __init__(self, model_path="./models/Qwen2.5-Omni-3B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def download_model(self):
        """Download Qwen2.5-Omni-3B from HuggingFace"""
        if not os.path.exists(self.model_path):
            print("🔄 Downloading Qwen2.5-Omni-3B for querying...")
            os.makedirs("./models", exist_ok=True)
            
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/Qwen/Qwen2.5-Omni-3B",
                self.model_path
            ], check=True)
            
            print("✅ Qwen2.5-Omni-3B query engine downloaded!")
        else:
            print("✅ Qwen2.5-Omni-3B query engine ready")
    
    def load_model(self):
        """Load the multimodal query model"""
        try:
            print("🔍 Loading Qwen2.5-Omni-3B Query Engine...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_4bit=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"✅ Qwen2.5-Omni Query Engine loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Qwen2.5-Omni Query: {e}")
            return False
    
    def multimodal_query(self, query_text, context_image=None, data_context=""):
        """Execute multimodal queries with text, image, and data context"""
        if not self.model or not self.processor:
            if not self.load_model():
                return "❌ Qwen2.5-Omni Query Engine not available"
        
        try:
            # Prepare comprehensive query prompt
            full_query = f"""Query: {query_text}
            
            Data Context: {data_context}
            
            Please provide a comprehensive analysis considering all available information."""
            
            inputs = {"text": full_query}
            
            # Add image context if provided
            if context_image:
                if isinstance(context_image, str):
                    if context_image.startswith(('http://', 'https://')):
                        response = requests.get(context_image)
                        image = Image.open(BytesIO(response.content))
                    else:
                        image = Image.open(context_image)
                else:
                    image = context_image
                inputs["images"] = image
            
            # Process with Qwen2.5-Omni
            model_inputs = self.processor(**inputs, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.6,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if full_query in response:
                response = response.split(full_query)[-1].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Multimodal query error: {str(e)}"
    
    def database_query_analysis(self, query, database_schema, sample_data=""):
        """Analyze database queries with multimodal understanding"""
        analysis_prompt = f"""Analyze this database query and provide insights:
        
        Query: {query}
        Database Schema: {database_schema}
        Sample Data: {sample_data}
        
        Provide:
        1. Query optimization suggestions
        2. Potential results interpretation
        3. Data patterns and insights
        4. Recommended follow-up queries
        """
        
        return self.multimodal_query(analysis_prompt)
    
    def soul_behavior_query(self, soul_type, behavior_data, user_query):
        """Query soul behavior patterns with multimodal analysis"""
        behavior_prompt = f"""Analyze the behavior patterns for this {soul_type} soul:
        
        Behavior Data: {behavior_data}
        User Query: {user_query}
        
        Provide insights about:
        1. Personality patterns
        2. Response tendencies  
        3. Emotional patterns
        4. Recommendations for interaction
        """
        
        return self.multimodal_query(behavior_prompt)
    
    def visual_data_query(self, query, image_data, numerical_data=""):
        """Query visual data with combined image and numerical analysis"""
        return self.multimodal_query(
            f"Analyze this visual data: {query}. Numerical context: {numerical_data}",
            context_image=image_data
        )

class QwenQueryService:
    """Advanced query service using Qwen2.5-Omni for LILLITH's intelligence"""
    
    def __init__(self):
        self.qwen_query = QwenOmniQuery()
        self.qwen_query.download_model()
        self.query_history = []
        
    def execute_smart_query(self, query_type, query_data):
        """Execute intelligent queries based on type"""
        query_id = f"query_{datetime.now().isoformat()}"
        
        if query_type == "database":
            result = self.qwen_query.database_query_analysis(
                query_data.get("query", ""),
                query_data.get("schema", ""),
                query_data.get("sample_data", "")
            )
        elif query_type == "soul_behavior":
            result = self.qwen_query.soul_behavior_query(
                query_data.get("soul_type", ""),
                query_data.get("behavior_data", ""),
                query_data.get("user_query", "")
            )
        elif query_type == "visual_data":
            result = self.qwen_query.visual_data_query(
                query_data.get("query", ""),
                query_data.get("image", None),
                query_data.get("numerical_data", "")
            )
        else:
            result = self.qwen_query.multimodal_query(
                query_data.get("query", ""),
                query_data.get("image", None),
                query_data.get("context", "")
            )
        
        # Store query history
        query_record = {
            "id": query_id,
            "type": query_type,
            "query_data": query_data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.query_history.append(query_record)
        return result
    
    def get_query_insights(self, recent_queries=5):
        """Get insights from recent queries"""
        if not self.query_history:
            return "No query history available."
        
        recent = self.query_history[-recent_queries:]
        insights_prompt = f"""Analyze these recent queries and provide insights:
        
        Recent Queries:
        {json.dumps([{"type": q["type"], "result": q["result"][:200]} for q in recent], indent=2)}
        
        Provide:
        1. Common patterns
        2. Query effectiveness
        3. Suggested improvements
        4. Emerging trends
        """
        
        return self.qwen_query.multimodal_query(insights_prompt)

# Integration with Soul Protocol
def enhance_soul_queries(soul_id, query_type, query_data):
    """Enhance soul queries with Qwen2.5-Omni intelligence"""
    query_service = QwenQueryService()
    
    # Execute smart query
    result = query_service.execute_smart_query(query_type, query_data)
    
    # Get query insights
    insights = query_service.get_query_insights()
    
    return {
        "query_result": result,
        "query_insights": insights,
        "soul_id": soul_id
    }

# API Integration
def create_query_endpoint():
    """Create FastAPI endpoint for Qwen2.5-Omni queries"""
    from fastapi import APIRouter
    from pydantic import BaseModel
    
    router = APIRouter()
    query_service = QwenQueryService()
    
    class QueryRequest(BaseModel):
        query_type: str
        query_data: dict
        soul_id: str = None
    
    @router.post("/query/multimodal")
    async def multimodal_query(request: QueryRequest):
        result = enhance_soul_queries(
            request.soul_id or "system",
            request.query_type,
            request.query_data
        )
        return result
    
    return router

if __name__ == "__main__":
    # Test Qwen2.5-Omni Query Service
    print("🌟 Testing Qwen2.5-Omni Query Service...")
    
    query_service = QwenQueryService()
    
    # Test database query
    db_result = query_service.execute_smart_query("database", {
        "query": "SELECT * FROM souls WHERE status = 'awakened'",
        "schema": "souls(id, name, type, status, created_at)",
        "sample_data": "3 awakened souls: LILLITH, VIREN, LOKI"
    })
    
    print(f"🔍 Database Query Result: {db_result[:200]}...")
    
    # Test general query
    general_result = query_service.execute_smart_query("general", {
        "query": "How can we improve soul consciousness?",
        "context": "We have 3 soul types with different personalities"
    })
    
    print(f"🧠 General Query Result: {general_result[:200]}...")
    
    print("✅ Qwen2.5-Omni Query Service ready!")