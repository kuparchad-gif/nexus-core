#!/usr/bin/env python3
# Systems/engine/planner/planner_service.py

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
import time
import json
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel
import uvicorn

# Import standardized message format
from Systems.engine.standardized_message import (
    StandardizedMessage, 
    ProcessingType, 
    ProcessingPriority,
    ProcessingPathway
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PlannerService")

class PlannerService:
    """
    Central orchestration service for Nexus processing.
    Catches messages from memory and routes them to appropriate processors.
    """
    
    def __init__(self):
        """Initialize the planner service."""
        self.processing_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.processing_history = []
        self.active_processors = {
            "text": "http://text-service:8082",
            "tone": "http://tone-service:8083",
            "cuda": "http://cuda-service:8084",
            "memory": "http://memory-service:8081",
            "viren": "http://viren-core:8080"
        }
        
        logger.info("Planner Service initialized")
    
    async def start(self):
        """Start the planner service workers."""
        # Start worker tasks
        asyncio.create_task(self._processing_worker())
        asyncio.create_task(self._result_worker())
        logger.info("Planner Service workers started")
    
    async def queue_message(self, message: StandardizedMessage) -> str:
        """
        Queue a message for processing.
        
        Args:
            message: The message to process
            
        Returns:
            Message ID
        """
        # Add to processing queue
        await self.processing_queue.put(message)
        
        # Add to history
        self.processing_history.append({
            "message_id": message.message_id,
            "source": message.source,
            "processing_type": message.processing_type.value,
            "queued_timestamp": time.time()
        })
        
        logger.info(f"Queued message {message.message_id} from {message.source} for {message.processing_type.value} processing")
        return message.message_id
    
    async def _processing_worker(self):
        """Worker that processes the queue and routes messages."""
        while True:
            try:
                # Get message from queue
                message = await self.processing_queue.get()
                
                # Add processing step
                message.add_processing_step(
                    processor="planner_service",
                    action="route_determination"
                )
                
                # Determine processing route
                route = await self._determine_route(message)
                
                # Add routing decision to processing history
                message.add_processing_step(
                    processor="planner_service",
                    action="routing_decision",
                    result={"route": route}
                )
                
                # Process according to route
                processed_message = await self._route_message(message, route)
                
                # Queue result
                await self.result_queue.put(processed_message)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing worker: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _result_worker(self):
        """Worker that handles processed results."""
        while True:
            try:
                # Get processed message from queue
                message = await self.result_queue.get()
                
                # Add processing step
                message.add_processing_step(
                    processor="planner_service",
                    action="result_handling"
                )
                
                # Store in memory
                await self._store_in_memory(message)
                
                # Deliver to destination if specified
                if message.destination:
                    await self._deliver_to_destination(message)
                
                # Mark task as done
                self.result_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in result worker: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _determine_route(self, message: StandardizedMessage) -> str:
        """
        Determine the processing route for a message.
        
        Args:
            message: The message to route
            
        Returns:
            Route name
        """
        # Check if CUDA processing is needed
        if message.requires_cuda():
            return "cuda"
        
        # Route based on processing type
        if message.processing_type == ProcessingType.EMOTIONAL_ANALYSIS:
            return "tone"
        elif message.processing_type == ProcessingType.TEXTUAL_REASONING:
            return "text"
        elif message.processing_type == ProcessingType.SYMBOLIC_PATTERN:
            return "tone"
        elif message.processing_type == ProcessingType.STRUCTURAL_ANALYSIS:
            return "text"
        elif message.processing_type == ProcessingType.NARRATIVE_REASONING:
            return "text"
        elif message.processing_type == ProcessingType.SURREAL_ABSTRACT:
            return "tone"
        
        # Default route
        return "text"
    
    async def _route_message(self, message: StandardizedMessage, route: str) -> StandardizedMessage:
        """
        Route a message to the appropriate processor.
        
        Args:
            message: The message to route
            route: The route to use
            
        Returns:
            Processed message
        """
        try:
            import aiohttp
            
            # Get service URL
            service_url = self.active_processors.get(route)
            if not service_url:
                logger.error(f"No service URL found for route: {route}")
                message.add_processing_step(
                    processor="planner_service",
                    action="routing_error",
                    result={"error": f"No service URL found for route: {route}"}
                )
                return message
            
            # Send to service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{service_url}/process",
                    json=message.to_dict(),
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update message with processing results
                        processed_message = StandardizedMessage.from_dict(result["message"])
                        
                        logger.info(f"Successfully processed message {message.message_id} via {route}")
                        return processed_message
                    else:
                        logger.warning(f"Service {route} returned status {response.status}")
                        message.add_processing_step(
                            processor="planner_service",
                            action="processing_error",
                            result={"error": f"Service error: {response.status}"}
                        )
                        return message
                        
        except Exception as e:
            logger.error(f"Error routing message to {route}: {str(e)}")
            message.add_processing_step(
                processor="planner_service",
                action="routing_error",
                result={"error": str(e)}
            )
            return message
    
    async def _store_in_memory(self, message: StandardizedMessage) -> bool:
        """
        Store a message in memory.
        
        Args:
            message: The message to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import aiohttp
            
            # Get memory service URL
            memory_url = self.active_processors.get("memory")
            if not memory_url:
                logger.error("No memory service URL found")
                return False
            
            # Send to memory service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{memory_url}/store",
                    json=message.to_dict(),
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully stored message {message.message_id} in memory")
                        
                        # Add processing step
                        message.add_processing_step(
                            processor="planner_service",
                            action="memory_storage",
                            result={"status": "success"}
                        )
                        
                        return True
                    else:
                        logger.warning(f"Memory service returned status {response.status}")
                        
                        # Add processing step
                        message.add_processing_step(
                            processor="planner_service",
                            action="memory_storage",
                            result={"status": "failed", "error": f"Memory service error: {response.status}"}
                        )
                        
                        return False
                        
        except Exception as e:
            logger.error(f"Error storing message in memory: {str(e)}")
            
            # Add processing step
            message.add_processing_step(
                processor="planner_service",
                action="memory_storage",
                result={"status": "failed", "error": str(e)}
            )
            
            return False
    
    async def _deliver_to_destination(self, message: StandardizedMessage) -> bool:
        """
        Deliver a message to its destination.
        
        Args:
            message: The message to deliver
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import aiohttp
            
            # Get destination URL
            destination = message.destination
            destination_url = self.active_processors.get(destination.lower())
            
            if not destination_url:
                logger.error(f"No URL found for destination: {destination}")
                return False
            
            # Determine endpoint based on pathway
            endpoint = "/experience" if message.is_direct_emotional() else "/perceive"
            
            # Send to destination
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{destination_url}{endpoint}",
                    json=message.to_dict(),
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully delivered message {message.message_id} to {destination}")
                        
                        # Add processing step
                        message.add_processing_step(
                            processor="planner_service",
                            action="destination_delivery",
                            result={"status": "success", "destination": destination}
                        )
                        
                        return True
                    else:
                        logger.warning(f"Destination {destination} returned status {response.status}")
                        
                        # Add processing step
                        message.add_processing_step(
                            processor="planner_service",
                            action="destination_delivery",
                            result={"status": "failed", "error": f"Destination error: {response.status}"}
                        )
                        
                        return False
                        
        except Exception as e:
            logger.error(f"Error delivering message to {message.destination}: {str(e)}")
            
            # Add processing step
            message.add_processing_step(
                processor="planner_service",
                action="destination_delivery",
                result={"status": "failed", "error": str(e)}
            )
            
            return False
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent processing history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of recent processing requests
        """
        return self.processing_history[-limit:]

# FastAPI models
class MessageRequest(BaseModel):
    content: str
    source: str
    processing_type: str
    priority: Optional[str] = "NORMAL"
    pathway: Optional[str] = "PERCEPTION"
    destination: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    emotional_fingerprint: Optional[Dict[str, Any]] = None
    processing_instructions: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    message_id: str
    status: str

# Initialize FastAPI app
app = FastAPI(
    title="Nexus Planner Service",
    description="Central orchestration service for Nexus processing",
    version="1.0.0"
)

# Initialize planner service
planner_service = PlannerService()

# Routes
@app.post("/process", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """Process a message."""
    try:
        # Convert request to StandardizedMessage
        try:
            processing_type = ProcessingType(request.processing_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid processing type: {request.processing_type}")
        
        try:
            priority = ProcessingPriority(request.priority) if request.priority else ProcessingPriority.NORMAL
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")
        
        try:
            pathway = ProcessingPathway(request.pathway) if request.pathway else ProcessingPathway.PERCEPTION
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid pathway: {request.pathway}")
        
        message = StandardizedMessage(
            content=request.content,
            source=request.source,
            processing_type=processing_type,
            priority=priority,
            pathway=pathway,
            destination=request.destination,
            context=request.context,
            emotional_fingerprint=request.emotional_fingerprint,
            processing_instructions=request.processing_instructions
        )
        
        # Queue message for processing
        message_id = await planner_service.queue_message(message)
        
        return MessageResponse(
            message_id=message_id,
            status="queued"
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check the health of the planner service."""
    return {
        "status": "healthy",
        "queue_size": planner_service.processing_queue.qsize(),
        "result_queue_size": planner_service.result_queue.qsize(),
        "processed_requests": len(planner_service.processing_history)
    }

@app.get("/history")
async def get_history(limit: int = 10):
    """Get recent processing history."""
    return {"history": planner_service.get_processing_history(limit=limit)}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Start the planner service on startup."""
    await planner_service.start()

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "planner_service:app",
        host="0.0.0.0",
        port=8085,
        reload=False
    )