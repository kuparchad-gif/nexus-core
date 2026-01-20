#!/usr/bin/env python3
"""
LLM Bridge Endpoint
Enables communication between different LLM models
"""
from fastapi import APIRouter, HTTPException
import asyncio
import json
import os
from typing import Dict, Any

# Create router
router = APIRouter()

@router.post("/bridge")
async def bridge_llms(request: Dict[str, Any]):
    """
    Bridge between two LLM models
    Translates messages between different model contexts
    """
    source_model = request.get("source")
    target_model = request.get("target")
    message = request.get("message")
    
    if not source_model or not target_model or not message:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    # Get LLM managers for source and target
    source_llm = get_llm_manager(source_model)
    target_llm = get_llm_manager(target_model)
    
    # Create bridge prompt
    bridge_prompt = f"""
You are translating a message from {source_model} to {target_model}.
Adapt this message to be optimally understood by {target_model} while preserving the core meaning.

Original message: {message}

Translated message for {target_model}:
"""
    
    # Generate bridge translation
    bridge_result = await source_llm.generate(bridge_prompt, max_tokens=256)
    
    # Process with target model
    target_prompt = f"""
You are {target_model} receiving a message originally from {source_model}.
The message has been translated for your understanding.

Message: {bridge_result.get('text', message)}

How do you respond?
"""
    
    target_response = await target_llm.generate(target_prompt, max_tokens=256)
    
    return {
        "source_model": source_model,
        "target_model": target_model,
        "original_message": message,
        "bridged_message": bridge_result.get("text", message),
        "target_response": target_response.get("text", "")
    }

def get_llm_manager(model_name: str):
    """Get LLM manager for a specific model"""
    from llm_manager import LLMManager
    return LLMManager(api_key=os.environ.get("HF_TOKEN"))