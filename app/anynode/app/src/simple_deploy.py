#!/usr/bin/env python3
"""
Modal Deployment for Aethereal Nexus
Deploys a model to Modal's infrastructure
"""

import os
import modal

# Create Modal stub
stub = modal.Stub("aethereal-nexus")

# Create base image
base_image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "huggingface_hub"
)

# Create model function at global scope
@stub.function(
    image=base_image,
    gpu="T4",
    memory=16384,
    timeout=600,
    concurrency_limit=10
)
async def gemma_model(prompt: str, **kwargs):
    """Modal function to run inference with Gemma 3B model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_id = "google/gemma-3-3b-it"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("temperature", 0.7) > 0
        )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "model_id": model_id,
        "output": output_text,
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
            "total_tokens": outputs.shape[1]
        }
    }

if __name__ == "__main__":
    # Deploy the application
    app.deploy()
    print("Aethereal Nexus deployed successfully!")
    print("You can now use the model at: https://aethereal-nexus--gemma-model.modal.run")
