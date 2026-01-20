#!/usr/bin/env python3
# Systems/engine/tone/tone_integration.py

import asyncio
import logging
from typing import Dict, Any, Optional, List
import json

from .tone_processor import ToneProcessor, ProcessingMode
from .tone_memory_uploader import ToneMemoryUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ToneIntegration")

class ToneIntegration:
    """
    Integration layer that connects the tone processor with other Nexus components.
    Handles the flow of tone processing and memory storage.
    """
    
    def __init__(self, 
                memory_service_url: str = "http://memory-service:8081",
                text_service_url: str = "http://text-service:8082"):
        """
        Initialize the tone integration layer.
        
        Args:
            memory_service_url: URL of the memory service
            text_service_url: URL of the text service
        """
        self.tone_processor = ToneProcessor()
        self.memory_uploader = ToneMemoryUploader(memory_service_url=memory_service_url)
        self.text_service_url = text_service_url
        
        logger.info("Tone Integration layer initialized")
    
    async def start(self):
        """Start the integration services."""
        await self.memory_uploader.start_uploader()
        logger.info("Tone Integration services started")
    
    async def process_and_store(self, 
                          text: str, 
                          mode: Optional[ProcessingMode] = None,
                          context: Optional[Dict[str, Any]] = None,
                          store_memory: bool = True,
                          analyze_text: bool = True) -> Dict[str, Any]:
        """
        Process text tone and store the results in memory.
        
        Args:
            text: The text to process
            mode: Processing mode to use
            context: Additional context for processing
            store_memory: Whether to store results in memory
            analyze_text: Whether to also analyze text content
            
        Returns:
            Dictionary with processing results and metadata
        """
        # Process tone
        tone_result = await self.tone_processor.process_text(
            text=text,
            mode=mode,
            context=context
        )
        
        # Create response object
        response = {
            "tone_analysis": tone_result,
            "processing_id": f"tone_{int(asyncio.get_event_loop().time())}"
        }
        
        # Store in memory if requested
        if store_memory:
            upload_id = await self.memory_uploader.queue_upload(
                text_data={"text": text, "context": context},
                tone_result=tone_result
            )
            response["memory_upload_id"] = upload_id
        
        # Analyze text if requested
        if analyze_text:
            text_result = await self._analyze_text(text)
            response["text_analysis"] = text_result
        
        return response
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Send text to text service for analysis.
        
        Args:
            text: The text to analyze
            
        Returns:
            Text analysis results
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.text_service_url}/process",
                    json={"text": text},
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("result", {})
                    else:
                        logger.warning(f"Text service returned status {response.status}")
                        return {"error": f"Text service error: {response.status}"}
        except Exception as e:
            logger.error(f"Error calling text service: {str(e)}")
            return {"error": f"Failed to analyze text: {str(e)}"}
    
    async def process_batch(self, texts: List[str], mode: Optional[ProcessingMode] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of texts for tone analysis.
        
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
    
    async def create_emotional_fingerprint(self, text: str) -> Dict[str, Any]:
        """
        Create an emotional fingerprint for a text.
        This is a specialized function for memory tagging.
        
        Args:
            text: The text to analyze
            
        Returns:
            Emotional fingerprint data
        """
        # Process with emotional analysis mode
        result = await self.tone_processor.process_text(
            text=text,
            mode=ProcessingMode.EMOTIONAL_ANALYSIS
        )
        
        # Create fingerprint
        fingerprint = {
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "intensity": result.get("intensity", 0),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Add emotion distribution if available
        if "emotion_distribution" in result:
            fingerprint["emotion_distribution"] = result["emotion_distribution"]
        
        return fingerprint
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get the status of the memory uploader.
        
        Returns:
            Dictionary with memory uploader status
        """
        return self.memory_uploader.get_queue_status()

# Example usage
async def example_usage():
    integration = ToneIntegration()
    await integration.start()
    
    # Process a single text
    result = await integration.process_and_store(
        "I am incredibly happy about this wonderful news! It's the most exciting thing that has happened all year."
    )
    
    print("Processing Result:", json.dumps(result, indent=2))
    
    # Process a batch of texts
    batch_results = await integration.process_batch([
        "The journey to consciousness begins with memory.",
        "I feel deeply moved by the beauty of this creation."
    ])
    
    print(f"Processed {len(batch_results)} texts in batch")
    
    # Create emotional fingerprint
    fingerprint = await integration.create_emotional_fingerprint(
        "This fills me with hope and joy for the future."
    )
    
    print("Emotional Fingerprint:", json.dumps(fingerprint, indent=2))
    
    # Check memory status
    print("Memory Status:", integration.get_memory_status())

if __name__ == "__main__":
    asyncio.run(example_usage())