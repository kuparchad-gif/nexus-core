#!/usr/bin/env python3
"""
Hyperbolic Image Processor for Cloud Viren
Handles image processing through Hyperbolic platform
"""

import os
import json
import time
import base64
import requests
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HyperbolicImageProcessor")

class HyperbolicImageProcessor:
    """
    Image processor using Hyperbolic platform
    Handles image analysis, generation, and manipulation
    """
    
    def __init__(self, api_key: str = None, config_path: str = None):
        """Initialize the Hyperbolic image processor"""
        self.api_key = api_key or os.environ.get("HYPERBOLIC_API_KEY")
        self.config_path = config_path or os.path.join("config", "hyperbolic_config.json")
        self.base_url = "https://api.hyperbolic.ai/v1"  # Example URL
        self.models = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load Hyperbolic configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key") or self.api_key
                    self.base_url = config.get("base_url") or self.base_url
                    self.models = config.get("models", {})
                    logger.info("Loaded Hyperbolic configuration")
            except Exception as e:
                logger.error(f"Error loading Hyperbolic configuration: {e}")
    
    def _save_config(self):
        """Save Hyperbolic configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            config = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "models": self.models
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Saved Hyperbolic configuration")
        except Exception as e:
            logger.error(f"Error saving Hyperbolic configuration: {e}")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image as base64
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _decode_image(self, base64_string: str) -> Image.Image:
        """
        Decode base64 string to image
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            PIL Image object
        """
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    def analyze_image(self, image_path: str, model_id: str = "vision-default") -> Dict[str, Any]:
        """
        Analyze an image using Hyperbolic vision models
        
        Args:
            image_path: Path to image file
            model_id: Vision model ID
            
        Returns:
            Dictionary with analysis results
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Analyzing image with model {model_id}")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_id,
                "image": base64_image,
                "analysis_type": "full"
            }
            
            # Send request for image analysis
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/vision/analyze",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_time = time.time() - start_time
                
                logger.info(f"Image analyzed successfully in {analysis_time:.2f} seconds")
                return {
                    "status": "success",
                    "analysis": result.get("analysis", {}),
                    "analysis_time": analysis_time,
                    "model_id": model_id
                }
            else:
                logger.error(f"Error analyzing image: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error analyzing image: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_image(self, prompt: str, width: int = 512, height: int = 512,
                      model_id: str = "stable-diffusion-xl") -> Dict[str, Any]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            model_id: Image generation model ID
            
        Returns:
            Dictionary with generated image data
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Generating image with model {model_id}")
        
        try:
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_id,
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_images": 1
            }
            
            # Send request for image generation
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/images/generations",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                generation_time = time.time() - start_time
                
                # Get image data
                image_data = result.get("data", [{}])[0].get("b64_json", "")
                
                logger.info(f"Image generated successfully in {generation_time:.2f} seconds")
                return {
                    "status": "success",
                    "image_data": image_data,
                    "generation_time": generation_time,
                    "model_id": model_id
                }
            else:
                logger.error(f"Error generating image: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error generating image: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {"status": "error", "message": str(e)}
    
    def image_to_text(self, image_path: str, prompt: str = "Describe this image in detail.",
                     model_id: str = "llava-1.5-7b") -> Dict[str, Any]:
        """
        Convert image to text description
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for the model
            model_id: Vision-language model ID
            
        Returns:
            Dictionary with text description
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Converting image to text with model {model_id}")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_id,
                "image": base64_image,
                "prompt": prompt,
                "max_tokens": 1024
            }
            
            # Send request for image-to-text conversion
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/vision/generate",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                text = result.get("text", "")
                
                logger.info(f"Image converted to text successfully in {processing_time:.2f} seconds")
                return {
                    "status": "success",
                    "text": text,
                    "processing_time": processing_time,
                    "model_id": model_id
                }
            else:
                logger.error(f"Error converting image to text: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error converting image to text: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error converting image to text: {e}")
            return {"status": "error", "message": str(e)}
    
    def process_image(self, image_path: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process an image with multiple operations
        
        Args:
            image_path: Path to image file
            operations: List of operations to perform
            
        Returns:
            Dictionary with processed image data
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Processing image with {len(operations)} operations")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image": base64_image,
                "operations": operations
            }
            
            # Send request for image processing
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/images/process",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                # Get processed image data
                processed_image = result.get("processed_image", "")
                
                logger.info(f"Image processed successfully in {processing_time:.2f} seconds")
                return {
                    "status": "success",
                    "processed_image": processed_image,
                    "processing_time": processing_time,
                    "operations_applied": operations
                }
            else:
                logger.error(f"Error processing image: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error processing image: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"status": "error", "message": str(e)}
    
    def detect_objects(self, image_path: str, confidence: float = 0.5,
                      model_id: str = "yolov8") -> Dict[str, Any]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to image file
            confidence: Minimum confidence threshold
            model_id: Object detection model ID
            
        Returns:
            Dictionary with detected objects
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info(f"Detecting objects with model {model_id}")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_id,
                "image": base64_image,
                "confidence": confidence
            }
            
            # Send request for object detection
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/vision/detect",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                detection_time = time.time() - start_time
                
                # Get detected objects
                objects = result.get("objects", [])
                
                logger.info(f"Objects detected successfully in {detection_time:.2f} seconds")
                return {
                    "status": "success",
                    "objects": objects,
                    "detection_time": detection_time,
                    "model_id": model_id,
                    "object_count": len(objects)
                }
            else:
                logger.error(f"Error detecting objects: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error detecting objects: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get available Hyperbolic models
        
        Returns:
            Dictionary with available models
        """
        if not self.api_key:
            return {"status": "error", "message": "API key not set"}
        
        logger.info("Getting available models")
        
        try:
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Send request for available models
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update models cache
                self.models = result.get("data", [])
                self._save_config()
                
                logger.info(f"Retrieved {len(self.models)} available models")
                return {
                    "status": "success",
                    "models": self.models
                }
            else:
                logger.error(f"Error getting models: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Error getting models: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return {"status": "error", "message": str(e)}

# Example usage
if __name__ == "__main__":
    # Create Hyperbolic image processor
    processor = HyperbolicImageProcessor()
    
    # Example image path
    image_path = "test_image.jpg"
    
    # Analyze image
    if os.path.exists(image_path):
        result = processor.analyze_image(image_path)
        print(f"Analysis result: {result}")
        
        # Generate image
        result = processor.generate_image("A futuristic city with flying cars")
        print(f"Generation result: {result}")
        
        # Convert image to text
        result = processor.image_to_text(image_path)
        print(f"Image to text result: {result}")
    else:
        print(f"Image {image_path} not found")