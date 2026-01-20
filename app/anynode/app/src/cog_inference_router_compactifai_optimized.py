import os
import asyncio
import logging
import json
from typing import Dict, Optional, List
from llama_cpp import Llama
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from datetime import datetime, timedelta
from transformers import AutoTokenizer
import qdrant_client  # Added import
from nats.aio.client import Client as NATS
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
import psutil
import torch  # Added import

# ... [rest of imports and config]

# Enhanced ModelManager with unloading
class ModelManager:
    _models = {}
    _tokenizers = {}
    _executor = ThreadPoolExecutor(max_workers=Config.THREADS, thread_name_prefix="inference_worker")
    _model_usage = {}

    @classmethod
    def get_model(cls, model_path: str) -> Llama:
        if model_path not in cls._models:
            logger.info(f"Loading model from {model_path}")
            cls._models[model_path] = Llama(
                model_path=model_path, 
                n_threads=Config.THREADS, 
                n_ctx=4096, 
                n_batch=512
            )
            cls._model_usage[model_path] = datetime.now()
        else:
            cls._model_usage[model_path] = datetime.now()
        return cls._models[model_path]

    @classmethod
    def unload_unused_models(cls, max_age_minutes: int = 60):
        """Unload models not used recently to free memory"""
        now = datetime.now()
        for model_path, last_used in list(cls._model_usage.items()):
            if (now - last_used).total_seconds() > max_age_minutes * 60:
                logger.info(f"Unloading model {model_path}")
                del cls._models[model_path]
                del cls._model_usage[model_path]
                if model_path in cls._tokenizers:
                    del cls._tokenizers[model_path]

# Enhanced SmartCache with better memory management
class SmartCache:
    def __init__(self, max_size: int, ttl_hours: float, max_memory: int):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.max_memory = max_memory
        self.current_memory = 0

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            value, timestamp, size = self.cache[key]
            if datetime.now() - timestamp > self.ttl:
                self.current_memory -= size
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return value
        return None

    def set(self, key: str, value: str):
        size = len(value.encode('utf-8'))
        if size > self.max_memory:
            return  # Skip if value is too large
        
        if len(self.cache) >= self.max_size or self.current_memory + size > self.max_memory:
            self._evict_oldest()
        
        self.cache[key] = (value, datetime.now(), size)
        self.current_memory += size
        self.cache.move_to_end(key)

    def _evict_oldest(self):
        while self.cache and (len(self.cache) >= self.max_size or self.current_memory > self.max_memory):
            key, (_, _, size) = self.cache.popitem(last=False)
            self.current_memory -= size

# Improved complexity estimation
async def estimate_complexity(tokenizer, model, text: str) -> float:
    """Estimate query complexity using a heuristic approach"""
    # Count tokens as a simple complexity measure
    token_count = len(tokenizer.encode(text))
    
    # Check for complex keywords
    complex_keywords = {"analyze", "compare", "evaluate", "explain", "predict", "what if"}
    keyword_count = sum(1 for word in complex_keywords if word in text.lower())
    
    # Normalize to 0-1 range
    complexity = min(1.0, (token_count / 1000) + (keyword_count * 0.1))
    return complexity

# Enhanced NATS connection handling
async def setup_nats():
    nc = NATS()
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            await nc.connect(servers=[Config.NATS_URL])
            logger.info("Connected to NATS")
            return nc
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"Failed to connect to NATS after {max_attempts} attempts: {e}")
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Enhanced routing logic
async def route_request(request_text: str, user_id: str) -> Dict:
    with REQUEST_LATENCY.labels(model="total").time():
        REQUEST_COUNT.labels(model="total").inc()
        context = await fetch_context(user_id)
        enriched_input = f"{request_text} [CONTEXT: {context}]"
        cache_key = f"{user_id}:{request_text}"

        cached = stm_cache.get(cache_key)
        if cached:
            CACHE_HITS.inc()
            return {"response": cached, "latency": "instant", "model": "cached"}

        tokenizer = ModelManager.get_tokenizer(Config.MODEL_PATH_1B)
        complexity = await estimate_complexity(tokenizer, ModelManager.get_model(Config.MODEL_PATH_1B), request_text)

        if complexity < 0.3 or any(kw in request_text.lower() for kw in {"simple", "Q&A", "classification", "summarize"}):
            model_path = Config.MODEL_PATH_1B
            latency = "ms"
        elif complexity < 0.7 and await viren_governance_check(request_text):
            model_path = Config.MODEL_PATH_7B
            latency = "100ms"
        else:
            if await viren_governance_check(request_text) and await check_budget():
                model_path = Config.MODEL_PATH_180B
                tokens = tokenizer(enriched_input, return_tensors="pt").input_ids[0].tolist()
                outputs = await ModelManager.infer_async(ModelManager.get_model(model_path), tokens)
                response = outputs['choices'][0]['text']
                stm_cache.set(cache_key, response)
                # Store in Qdrant
                try:
                    qdrant.upsert(
                        collection_name="inference_cache", 
                        points=[{
                            "id": cache_key, 
                            "vector": tokenizer.encode(response)[:128],  # Truncate to 128 dim
                            "payload": {"response": response, "user_id": user_id}
                        }]
                    )
                except Exception as e:
                    logger.error(f"Failed to store in Qdrant: {e}")
                REQUEST_LATENCY.labels(model="falcon-180b").observe(30)
                return {"response": response, "latency": "30s", "model": "falcon-180b"}
            return {"error": "Request rejected", "route": "7b"}

        tokens = tokenizer(enriched_input, return_tensors="pt").input_ids[0].tolist()
        outputs = await ModelManager.infer_async(ModelManager.get_model(model_path), tokens)
        response = outputs['choices'][0]['text']
        stm_cache.set(cache_key, response)
        
        model_name = "falcon-1b" if model_path == Config.MODEL_PATH_1B else "falcon-7b"
        REQUEST_LATENCY.labels(model=model_name).observe(float(latency.strip("ms")) / 1000)
        return {"response": response, "latency": latency, "model": model_name}

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Add model cleanup task
@app.on_event("startup")
async def startup():
    # ... existing code ...
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # Run every hour
        ModelManager.unload_unused_models()