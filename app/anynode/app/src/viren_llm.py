import modal
import os

# Set Modal profile
os.system("modal config set profile aethereal-nexus")

# Create Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install([
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "fastapi",
    "uvicorn",
    "pillow",
    "accelerate",
    "bitsandbytes"
])

# VIREN LLM Services
app = modal.App("viren-llm", image=image)

@app.function(cpu=4.0, memory=8192)
@modal.asgi_app()
def llm_server():
    from fastapi import FastAPI, Request
    from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
    import base64
    from io import BytesIO
    from PIL import Image
    import torch
    
    fast_app = FastAPI()
    
    # Load DialoGPT model (open, not gated)
    dialog_model_id = "microsoft/DialoGPT-medium"
    dialog_model = AutoModelForCausalLM.from_pretrained(
        dialog_model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    dialog_tokenizer = AutoTokenizer.from_pretrained(dialog_model_id)
    
    # Load BLIP model (open, not gated)
    blip_model_id = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(blip_model_id)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    @fast_app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        decoding = body.get("decoding", {})
        model_type = body.get("model", "dialog")  # Default to DialoGPT
        
        if model_type == "dialog":
            # Format prompt for DialoGPT
            inputs = dialog_tokenizer.encode(prompt + dialog_tokenizer.eos_token, return_tensors="pt").to(dialog_model.device)
            
            # Dynamic decoding parameters
            output = dialog_model.generate(
                inputs,
                max_new_tokens=decoding.get("max_new_tokens", 512),
                temperature=decoding.get("temperature", 0.7),
                top_p=decoding.get("top_p", 0.9),
                top_k=decoding.get("top_k", 50),
                num_beams=decoding.get("num_beams", 1),
                do_sample=decoding.get("do_sample", True),
                pad_token_id=dialog_tokenizer.eos_token_id
            )
            
            return {"output": dialog_tokenizer.decode(output[0], skip_special_tokens=True)}
            
        elif model_type == "blip":
            # Process image if provided
            image_data = body.get("image")
            if not image_data:
                return {"error": "Image required for BLIP model"}
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                return {"error": f"Invalid image data: {str(e)}"}
            
            # Process inputs
            inputs = blip_processor(image, prompt, return_tensors="pt").to(blip_model.device)
            
            # Generate
            output = blip_model.generate(
                **inputs,
                max_new_tokens=decoding.get("max_new_tokens", 256),
                temperature=decoding.get("temperature", 0.7),
                do_sample=decoding.get("do_sample", True)
            )
            
            return {"output": blip_processor.decode(output[0], skip_special_tokens=True)}
        
        else:
            return {"error": "Invalid model type. Use 'dialog' or 'blip'"}
    
    @fast_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "viren_llm"}
    
    return fast_app

@app.function(cpu=2.0, memory=4096)
@modal.asgi_app()
def tinyllama_server():
    from fastapi import FastAPI, Request
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    fast_app = FastAPI()
    
    # Load TinyLlama model - open model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    @fast_app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.2)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        result = response[len(prompt):].strip()
        
        return {"output": result}
    
    @fast_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "tinyllama"}
    
    return fast_app

if __name__ == "__main__":
    modal.run(app)