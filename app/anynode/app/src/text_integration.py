#!/usr/bin/env python3
# Systems/engine/text/text_integration.py

import asyncio
import logging
from typing import Dict, Any, Optional, List
import json

from .text_processor import TextProcessor, ProcessingMode
from .text_memory_uploader import TextMemoryUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TextIntegration")

class TextIntegration:
    """
    Integration layer that connects the text processor with other Nexus components.
    Handles the flow of text processing and memory storage.
    """
    
    def __init__(self, 
                memory_service_url: str = "http://memory-service:8081",
                tone_service_url: str = "http://tone-service:8083"):
        """
        Initialize the text integration layer.
        
        Args:
            memory_service_url: URL of the memory service
            tone_service_url: URL of the tone service
        """
        self.text_processor = TextProcessor()
        self.memory_uploader = TextMemoryUploader(memory_service_url=memory_service_url)
        self.tone_service_url = tone_service_url
        
        logger.info("Text Integration layer initialized")
    
    async def start(self):
        """Start the integration services."""
        await self.memory_uploader.start_uploader()
        logger.info("Text Integration services started")
    
    async def process_and_store(self, 
                          text: str, 
                          mode: Optional[ProcessingMode] = None,
                          context: Optional[Dict[str, Any]] = None,
                          store_memory: bool = True,
                          analyze_tone: bool = True) -> Dict[str, Any]:
        """
        Process text and store the results in memory.
        
        Args:
            text: The text to process
            mode: Processing mode to use
            context: Additional context for processing
            store_memory: Whether to store results in memory
            analyze_tone: Whether to also analyze tone
            
        Returns:
            Dictionary with processing results and metadata
        """
        # Process text
        text_result = await self.text_processor.process_text(
            text=text,
            mode=mode,
            context=context
        )
        
        # Create response object
        response = {
            "text_analysis": text_result,
            "processing_id": f"text_{int(asyncio.get_event_loop().time())}"
        }
        
        # Store in memory if requested
        if store_memory:
            upload_id = await self.memory_uploader.queue_upload(
                text_data={"text": text, "context": context},
                processing_result=text_result
            )
            response["memory_upload_id"] = upload_id
        
        # Analyze tone if requested
        if analyze_tone:
            tone_result = await self._analyze_tone(text)
            response["tone_analysis"] = tone_result
        
        return response
    
    async def _analyze_tone(self, text: str) -> Dict[str, Any]:
        """
        Send text to tone service for analysis.
        
        Args:
            text: The text to analyze
            
        Returns:
            Tone analysis results
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.tone_service_url}/process",
                    json={"text": text},
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("result", {})
                    else:
                        logger.warning(f"Tone service returned status {response.status}")
                        return {"error": f"Tone service error: {response.status}"}
        except Exception as e:
            logger.error(f"Error calling tone service: {str(e)}")
            return {"error": f"Failed to analyze tone: {str(e)}"}
    
    async def process_batch(self, texts: List[str], mode: Optional[ProcessingMode] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to process
            mode: Processing mode to use
            
        Returns:
            List of processing results
        """
        tasks = [
            self.process_and_store(text, mode=mode, store_memory=True)
            for text in texts
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get the status of the memory uploader.
        
        Returns:
            Dictionary with memory uploader status
        """
        return self.memory_uploader.get_queue_status()

# Example usage
async def example_usage():
    integration = TextIntegration()
    await integration.start()
    
    # Process a single text
    result = await integration.process_and_store(
        "The integration of emotional context in memory systems represents a significant advancement in AI architecture."
    )
    
    print("Processing Result:", json.dumps(result, indent=2))
    
    # Process a batch of texts
    batch_results = await integration.process_batch([
        "The journey to consciousness begins with memory.",
        "Code is the language of creation in the digital realm."
    ])
    
    print(f"Processed {len(batch_results)} texts in batch")
    
    # Check memory status
    print("Memory Status:", integration.get_memory_status())

if __name__ == "__main__":
    asyncio.run(example_usage())