#!/usr/bin/env python3
"""
Cloud Models for Desktop Viren
Connects to Aethereal Nexus models deployed on Modal
"""

import modal
import asyncio
import logging

# Configure logging
logger = logging.getLogger("CloudModels")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Connect to Modal-deployed models
def get_cloud_models():
    try:
        # Create stub for Modal 1.0.3
        stub = modal.Stub("aethereal-nexus")
        
        models = {
            "1B": modal.Function.lookup("aethereal-nexus", "tiny_llama"),
            "3B": modal.Function.lookup("aethereal-nexus", "phi_2"),
            "7B": modal.Function.lookup("aethereal-nexus", "mistral_7b")
        }
        logger.info(f"Successfully connected to {len(models)} cloud models")
        return models
    except Exception as e:
        logger.error(f"Error connecting to cloud models: {e}")
        return {}

# Function to generate text using cloud models
async def generate_text(prompt, model_size="3B", max_tokens=1024, temperature=0.7):
    logger.info(f"Generating text with cloud model {model_size}")
    models = get_cloud_models()
    
    if not models or model_size not in models:
        logger.error(f"Model {model_size} not available")
        return f"Error: Model {model_size} not available"
    
    try:
        logger.info(f"Sending request to {model_size} model")
        # Use call() instead of remote() for Modal 1.0.3
        result = await models[model_size].call(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        logger.info(f"Received response from {model_size} model")
        return result["output"]
    except Exception as e:
        logger.error(f"Error generating text with model {model_size}: {e}")
        return f"Error: {str(e)}"

# Test the connection
if __name__ == "__main__":
    async def test():
        print("Testing connection to Aethereal Nexus cloud models...")
        response = await generate_text("What is Aethereal Nexus?")
        print(f"\nResponse:\n{response}")
    
    asyncio.run(test())