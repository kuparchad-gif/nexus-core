import modal
from fastapi import FastAPI, Request
import torch

# Create Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers", "fastapi", "uvicorn", "torch", "accelerate", 
    "sentencepiece", "bitsandbytes"
)

app = modal.App("cloud-viren-sql", image=image)

@app.function(gpu="T4")
@modal.asgi_app()
def sql_model_server():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    fast_app = FastAPI()
    
    # Load TinyLlama SQL model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model, will be fine-tuned for SQL
    
    # For production, use the SQL-specific model
    # model_id = "tinyllama-coder-sql-en-v0.1"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    @fast_app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.1)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate SQL query
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True
        )
        
        # Extract response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        sql_query = response[len(prompt):].strip()
        
        return {"output": sql_query}
    
    return fast_app

if __name__ == "__main__":
    modal.runner.deploy_stub(app)