#!/usr/bin/env python3
# Systems/engine/tone/tone_service.py

import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn

from tone_processor import ToneProcessor, ProcessingMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ToneService")

# Initialize FastAPI app
app = FastAPI(
    title="Nexus Tone Service",
    description="Tone processing service for Viren/Eden",
    version="1.0.0"
)

# Initialize tone processor
tone_processor = ToneProcessor()

# Models for API
class ToneRequest(BaseModel):
    text: str
    mode: Optional[str] = "emotional_analysis"
    context: Optional[Dict[str, Any]] = None

class ToneResponse(BaseModel):
    result: Dict[str, Any]
    processing_time: float
    mode: str

class HealthResponse(BaseModel):
    status: str
    active_mode: str
    processed_requests: int

# Routes
@app.post("/process", response_model=ToneResponse)
async def process_text(request: ToneRequest):
    """Process text tone using the specified mode."""
    try:
        # Convert string mode to enum
        mode = None
        for m in ProcessingMode:
            if m.value == request.mode:
                mode = m
                break
        
        if not mode:
            raise HTTPException(status_code=400, detail=f"Invalid processing mode: {request.mode}")
        
        # Process the text
        start_time = asyncio.get_event_loop().time()
        result = await tone_processor.process_text(
            text=request.text,
            mode=mode,
            context=request.context
        )
        end_time = asyncio.get_event_loop().time()
        
        return ToneResponse(
            result=result,
            processing_time=end_time - start_time,
            mode=mode.value
        )
    except Exception as e:
        logger.error(f"Error processing tone: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the tone service."""
    return HealthResponse(
        status="healthy",
        active_mode=tone_processor.active_mode.value,
        processed_requests=len(tone_processor.processing_history)
    )

@app.get("/modes")
async def list_modes():
    """List available processing modes."""
    return {
        "modes": [mode.value for mode in ProcessingMode],
        "active_mode": tone_processor.active_mode.value
    }

@app.post("/mode/{mode}")
async def set_mode(mode: str):
    """Set the active processing mode."""
    try:
        # Convert string mode to enum
        mode_enum = None
        for m in ProcessingMode:
            if m.value == mode:
                mode_enum = m
                break
        
        if not mode_enum:
            raise HTTPException(status_code=400, detail=f"Invalid processing mode: {mode}")
        
        tone_processor.set_active_mode(mode_enum)
        return {"status": "success", "active_mode": mode}
    except Exception as e:
        logger.error(f"Error setting mode: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(limit: int = 10):
    """Get recent processing history."""
    return {"history": tone_processor.get_processing_history(limit=limit)}

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "tone_service:app",
        host="0.0.0.0",
        port=8083,
        reload=False
    )