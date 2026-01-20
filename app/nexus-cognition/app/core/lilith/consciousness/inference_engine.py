# C:\CogniKube-COMPLETE-FINAL\inference_engine.py
# Inference Engine - Tiny LLMs + Specialist LLMs for Query Processing

import modal
import os
import json
import time
import logging
import asyncio
import aiohttp
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
import numpy as np

# Modal configuration
image = modal.Image.debian_slim().pip_install([
    "fastapi==0.115.0",
    "uvicorn==0.30.6",
    "pydantic==2.9.2",
    "torch==2.1.0",
    "transformers==4.36.0",
    "numpy==1.24.3",
    "aiohttp==3.10.5"
])

app = modal.App("inference-engine", image=image)

# Configuration
TINY_LLM_COUNT = 3
SPECIALIST_DOMAINS = ["finance", "code", "medical", "troubleshooting"]

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class TinyLLM:
    def __init__(self, llm_id: str):
        self.llm_id = llm_id
        self.logger = setup_logger(f"tiny_llm_{llm_id}")
        self.is_polling = True
        self.poll_count = 0
        self.responses = []
        
    async def poll_query(self, query: str) -> Dict:
        """Always polling - provides quick insights"""
        try:
            self.poll_count += 1
            
            # Simulate tiny LLM processing (fast, lightweight)
            response = f"TinyLLM-{self.llm_id} insight: {query[:30]}... [quick analysis]"
            confidence = np.random.uniform(0.3, 0.7)  # Lower confidence
            
            result = {
                "llm_id": self.llm_id,
                "response": response,
                "confidence": confidence,
                "processing_time": 0.1,  # Very fast
                "poll_count": self.poll_count,
                "type": "polling_insight"
            }
            
            self.responses.append(result)
            
            self.logger.info({
                "action": "tiny_llm_polled",
                "llm_id": self.llm_id,
                "poll_count": self.poll_count
            })
            
            return result
            
        except Exception as e:
            self.logger.error({"action": "tiny_llm_poll_failed", "error": str(e)})
            return {
                "llm_id": self.llm_id,
                "error": str(e),
                "type": "polling_error"
            }

class SpecialistLLM:
    def __init__(self, specialty: str, weights_data: Optional[Dict] = None):
        self.specialty = specialty
        self.llm_id = f"specialist_{specialty}"
        self.logger = setup_logger(f"specialist_llm_{specialty}")
        self.weights_data = weights_data
        self.primary_role = "inference_polling"
        self.can_answer_when_polled = True
        self.query_count = 0
        self.poll_count = 0
        
    def load_weights(self, weights_data: Dict):
        """Load trained weights from training system"""
        self.weights_data = weights_data
        self.logger.info({
            "action": "weights_loaded",
            "specialty": self.specialty,
            "version": weights_data.get("version", "unknown")
        })
    
    async def poll_query(self, query: str) -> Dict:
        """Primary role: Polling for insights"""
        try:
            self.poll_count += 1
            
            # Check if query is relevant to specialty
            relevance = self.calculate_relevance(query)
            
            if relevance > 0.3:  # Relevant to specialty
                response = f"Specialist-{self.specialty}: {query[:50]}... [domain insight]"
                confidence = np.random.uniform(0.6, 0.9)  # Higher confidence in domain
            else:
                response = f"Specialist-{self.specialty}: Limited insight outside domain"
                confidence = np.random.uniform(0.2, 0.4)  # Lower confidence outside domain
            
            result = {
                "llm_id": self.llm_id,
                "specialty": self.specialty,
                "response": response,
                "confidence": confidence,
                "relevance": relevance,
                "processing_time": 0.3,
                "poll_count": self.poll_count,
                "type": "specialist_polling"
            }
            
            self.logger.info({
                "action": "specialist_polled",
                "specialty": self.specialty,
                "relevance": relevance
            })
            
            return result
            
        except Exception as e:
            self.logger.error({"action": "specialist_poll_failed", "error": str(e)})
            return {
                "llm_id": self.llm_id,
                "specialty": self.specialty,
                "error": str(e),
                "type": "specialist_error"
            }
    
    async def answer_query(self, query: str) -> Dict:
        """Secondary role: Can answer when directly polled"""
        if self.can_answer_when_polled:
            try:
                self.query_count += 1
                
                relevance = self.calculate_relevance(query)
                
                if relevance > 0.5:  # High relevance - provide full answer
                    response = f"Full {self.specialty} analysis: {query} [detailed specialist response]"
                    confidence = np.random.uniform(0.8, 0.95)
                else:
                    response = f"Limited {self.specialty} perspective: {query} [partial response]"
                    confidence = np.random.uniform(0.4, 0.6)
                
                result = {
                    "llm_id": self.llm_id,
                    "specialty": self.specialty,
                    "response": response,
                    "confidence": confidence,
                    "relevance": relevance,
                    "processing_time": 0.8,
                    "query_count": self.query_count,
                    "type": "specialist_answer"
                }
                
                self.logger.info({
                    "action": "specialist_answered",
                    "specialty": self.specialty,
                    "query_count": self.query_count
                })
                
                return result
                
            except Exception as e:
                self.logger.error({"action": "specialist_answer_failed", "error": str(e)})
                raise
        else:
            raise Exception(f"Specialist {self.specialty} cannot answer queries")
    
    def calculate_relevance(self, query: str) -> float:
        """Calculate how relevant query is to this specialty"""
        query_lower = query.lower()
        
        domain_keywords = {
            "finance": ["money", "stock", "trade", "investment", "bank", "finance", "market"],
            "code": ["code", "python", "javascript", "programming", "function", "class", "debug"],
            "medical": ["health", "medical", "doctor", "symptom", "treatment", "medicine"],
            "troubleshooting": ["error", "fix", "problem", "debug", "issue", "broken", "repair"]
        }
        
        keywords = domain_keywords.get(self.specialty, [])
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        
        return min(matches / len(keywords), 1.0) if keywords else 0.0

class InferenceOrchestrator:
    def __init__(self):
        self.logger = setup_logger("inference_orchestrator")
        
        # Initialize tiny LLMs (always polling)
        self.tiny_llms = [TinyLLM(f"tiny_{i}") for i in range(TINY_LLM_COUNT)]
        
        # Initialize specialist LLMs
        self.specialist_llms = {
            domain: SpecialistLLM(domain) 
            for domain in SPECIALIST_DOMAINS
        }
        
        self.inference_stats = {
            "total_queries": 0,
            "polling_responses": 0,
            "specialist_answers": 0,
            "combined_responses": 0
        }
    
    async def load_specialist_weights(self, specialty: str, weights_data: Dict):
        """Load trained weights into specialist LLM"""
        if specialty in self.specialist_llms:
            self.specialist_llms[specialty].load_weights(weights_data)
            self.logger.info({
                "action": "specialist_weights_loaded",
                "specialty": specialty
            })
        else:
            raise ValueError(f"Unknown specialty: {specialty}")
    
    async def poll_all_llms(self, query: str) -> Dict:
        """Poll all LLMs for insights (primary inference mode)"""
        try:
            self.inference_stats["total_queries"] += 1
            
            # Poll tiny LLMs (always active)
            tiny_responses = []
            for tiny_llm in self.tiny_llms:
                response = await tiny_llm.poll_query(query)
                tiny_responses.append(response)
            
            # Poll specialist LLMs
            specialist_responses = []
            for specialty, specialist_llm in self.specialist_llms.items():
                response = await specialist_llm.poll_query(query)
                specialist_responses.append(response)
            
            self.inference_stats["polling_responses"] += len(tiny_responses) + len(specialist_responses)
            
            # Combine insights
            combined_insight = self.combine_polling_responses(tiny_responses, specialist_responses)
            
            result = {
                "query": query,
                "tiny_llm_insights": tiny_responses,
                "specialist_insights": specialist_responses,
                "combined_insight": combined_insight,
                "total_polled": len(tiny_responses) + len(specialist_responses),
                "inference_type": "polling_mode"
            }
            
            self.logger.info({
                "action": "all_llms_polled",
                "tiny_responses": len(tiny_responses),
                "specialist_responses": len(specialist_responses)
            })
            
            return result
            
        except Exception as e:
            self.logger.error({"action": "poll_all_failed", "error": str(e)})
            raise
    
    async def query_specialist(self, query: str, specialty: str) -> Dict:
        """Query specific specialist LLM directly"""
        try:
            if specialty not in self.specialist_llms:
                raise ValueError(f"Unknown specialty: {specialty}")
            
            specialist_llm = self.specialist_llms[specialty]
            result = await specialist_llm.answer_query(query)
            
            self.inference_stats["specialist_answers"] += 1
            
            self.logger.info({
                "action": "specialist_queried",
                "specialty": specialty
            })
            
            return result
            
        except Exception as e:
            self.logger.error({"action": "query_specialist_failed", "error": str(e)})
            raise
    
    async def smart_inference(self, query: str, primary_llm_response: Optional[Dict] = None) -> Dict:
        """Combine smart LLM response with polling insights"""
        try:
            # Get polling insights
            polling_result = await self.poll_all_llms(query)
            
            # Combine with primary LLM response if provided
            if primary_llm_response:
                combined_response = {
                    "query": query,
                    "primary_llm_response": primary_llm_response,
                    "polling_insights": polling_result,
                    "inference_type": "smart_plus_polling",
                    "combined_at": datetime.now().isoformat()
                }
                
                self.inference_stats["combined_responses"] += 1
            else:
                combined_response = polling_result
            
            return combined_response
            
        except Exception as e:
            self.logger.error({"action": "smart_inference_failed", "error": str(e)})
            raise
    
    def combine_polling_responses(self, tiny_responses: List[Dict], specialist_responses: List[Dict]) -> Dict:
        """Combine polling responses into unified insight"""
        # Find highest confidence responses
        all_responses = tiny_responses + specialist_responses
        
        if not all_responses:
            return {"insight": "No responses available", "confidence": 0.0}
        
        # Get highest confidence response
        best_response = max(all_responses, key=lambda x: x.get("confidence", 0))
        
        # Calculate average confidence
        avg_confidence = np.mean([r.get("confidence", 0) for r in all_responses])
        
        return {
            "best_insight": best_response.get("response", ""),
            "best_llm": best_response.get("llm_id", "unknown"),
            "average_confidence": round(avg_confidence, 3),
            "total_insights": len(all_responses),
            "specialist_contributions": len(specialist_responses),
            "tiny_llm_contributions": len(tiny_responses)
        }
    
    def get_inference_stats(self) -> Dict:
        """Get inference engine statistics"""
        return {
            **self.inference_stats,
            "tiny_llms_active": len(self.tiny_llms),
            "specialist_llms_available": len(self.specialist_llms),
            "total_llms": len(self.tiny_llms) + len(self.specialist_llms)
        }

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    primary_llm_response: Optional[Dict] = None

class SpecialistQueryRequest(BaseModel):
    query: str
    specialty: str

class WeightsLoadRequest(BaseModel):
    specialty: str
    weights_data: Dict

@app.function(
    memory=4096,
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": "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"
    })]
)
@modal.asgi_app()
def inference_engine():
    """Inference Engine - Tiny LLMs + Specialist LLMs for Query Processing"""
    
    inference_app = FastAPI(title="Inference Engine - Polling & Answering")
    logger = setup_logger("inference_engine")
    orchestrator = InferenceOrchestrator()

    @inference_app.get("/")
    async def inference_status():
        """Inference engine status"""
        return {
            "system": "inference-engine",
            "status": "processing_queries",
            "purpose": "polling_and_answering_only",
            "tiny_llms": {
                "count": len(orchestrator.tiny_llms),
                "always_polling": True,
                "role": "quick_insights"
            },
            "specialist_llms": {
                "available": list(orchestrator.specialist_llms.keys()),
                "primary_role": "inference_polling",
                "can_answer": True
            },
            "inference_stats": orchestrator.get_inference_stats()
        }

    @inference_app.get("/health")
    async def health_check():
        """Health check for inference engine"""
        return {
            "system": "inference-engine",
            "status": "healthy",
            "tiny_llms_active": len(orchestrator.tiny_llms),
            "specialist_llms_ready": len(orchestrator.specialist_llms),
            "polling_active": True,
            "answering_ready": True
        }

    @inference_app.post("/poll")
    async def poll_all_llms(request: QueryRequest):
        """Poll all LLMs for insights"""
        try:
            result = await orchestrator.poll_all_llms(request.query)
            
            logger.info({
                "action": "poll_all_llms",
                "query_length": len(request.query),
                "total_polled": result["total_polled"]
            })
            
            return {
                "success": True,
                "polling_result": result
            }
            
        except Exception as e:
            logger.error({"action": "poll_all_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @inference_app.post("/query_specialist")
    async def query_specialist(request: SpecialistQueryRequest):
        """Query specific specialist LLM directly"""
        try:
            result = await orchestrator.query_specialist(request.query, request.specialty)
            
            logger.info({
                "action": "query_specialist",
                "specialty": request.specialty,
                "query_length": len(request.query)
            })
            
            return {
                "success": True,
                "specialist_response": result
            }
            
        except Exception as e:
            logger.error({"action": "query_specialist_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @inference_app.post("/smart_inference")
    async def smart_inference(request: QueryRequest):
        """Combine smart LLM response with polling insights"""
        try:
            result = await orchestrator.smart_inference(
                request.query, 
                request.primary_llm_response
            )
            
            logger.info({
                "action": "smart_inference",
                "has_primary_response": request.primary_llm_response is not None,
                "query_length": len(request.query)
            })
            
            return {
                "success": True,
                "smart_inference_result": result
            }
            
        except Exception as e:
            logger.error({"action": "smart_inference_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @inference_app.post("/load_weights")
    async def load_specialist_weights(request: WeightsLoadRequest):
        """Load trained weights into specialist LLM"""
        try:
            await orchestrator.load_specialist_weights(
                request.specialty, 
                request.weights_data
            )
            
            logger.info({
                "action": "weights_loaded",
                "specialty": request.specialty
            })
            
            return {
                "success": True,
                "specialty": request.specialty,
                "weights_loaded": True
            }
            
        except Exception as e:
            logger.error({"action": "load_weights_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

    @inference_app.get("/stats")
    async def inference_statistics():
        """Get inference engine statistics"""
        return {
            "success": True,
            "stats": orchestrator.get_inference_stats(),
            "system": "inference-engine"
        }

    return inference_app

if __name__ == "__main__":
    modal.run(app)