import modal
from fastapi import FastAPI, Request
import torch

# Create Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers", "fastapi", "uvicorn", "torch", "accelerate", 
    "sentencepiece", "bitsandbytes"
)

app = modal.App("cloud-viren-tinyllama", image=image)

@app.function(gpu="T4")
@modal.asgi_app()
def tinyllama_server():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    fast_app = FastAPI()
    
    # Load TinyLlama model
    model_id = "tinyllama-coder-en-v0.1"  # Use the specific model ID
    
    # Fallback to base model if specific one isn't available
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
    except:
        # Fallback to base TinyLlama
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
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
            do_sample=True
        )
        
        # Extract response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        result = response[len(prompt):].strip()
        
        return {"output": result}
    
    return fast_app

if __name__ == "__main__":
    modal.runner.deploy_stub(app)