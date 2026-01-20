import modal
from fastapi import FastAPI, Request
import torch

# Create Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers", "fastapi", "uvicorn", "torch", "accelerate", 
    "sentencepiece", "pillow", "bitsandbytes"
)

app = modal.App("cloud-viren-llm", image=image)

@app.function(gpu="A10G")
@modal.asgi_app()
def llm_server():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
    import base64
    from io import BytesIO
    from PIL import Image
    
    fast_app = FastAPI()
    
    # Load Mistral model
    mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral_model = AutoModelForCausalLM.from_pretrained(
        mistral_model_id, 
        torch_dtype=torch.float16, 
        device_map="auto",
        load_in_8bit=True
    )
    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
    
    # Load LLAVA model
    llava_model_id = "llava-hf/llava-1.5-7b-hf"
    llava_processor = AutoProcessor.from_pretrained(llava_model_id)
    llava_model = AutoModelForVision2Seq.from_pretrained(
        llava_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    @fast_app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        decoding = body.get("decoding", {})
        model_type = body.get("model", "mistral")  # Default to Mistral
        
        if model_type == "mistral":
            # Format prompt for Mistral
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            inputs = mistral_tokenizer(formatted_prompt, return_tensors="pt").to(mistral_model.device)
            
            # Dynamic decoding parameters
            output = mistral_model.generate(
                **inputs,
                max_new_tokens=decoding.get("max_new_tokens", 512),
                temperature=decoding.get("temperature", 0.7),
                top_p=decoding.get("top_p", 0.9),
                top_k=decoding.get("top_k", 50),
                num_beams=decoding.get("num_beams", 1),
                do_sample=decoding.get("do_sample", True)
            )
            
            return {"output": mistral_tokenizer.decode(output[0], skip_special_tokens=True)}
            
        elif model_type == "llava":
            # Process image if provided
            image_data = body.get("image")
            if not image_data:
                return {"error": "Image required for LLAVA model"}
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                return {"error": f"Invalid image data: {str(e)}"}
            
            # Process inputs
            inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(llava_model.device)
            
            # Generate
            output = llava_model.generate(
                **inputs,
                max_new_tokens=decoding.get("max_new_tokens", 256),
                temperature=decoding.get("temperature", 0.7),
                top_p=decoding.get("top_p", 0.9),
                do_sample=decoding.get("do_sample", True)
            )
            
            return {"output": llava_processor.decode(output[0], skip_special_tokens=True)}
        
        else:
            return {"error": "Invalid model type. Use 'mistral' or 'llava'"}
    
    return fast_app

if __name__ == "__main__":
    modal.runner.deploy_stub(app)