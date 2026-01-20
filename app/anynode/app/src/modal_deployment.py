import modal
import os

# Create Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Create a unified app for all Viren cloud services
app = modal.App("viren-cloud", image=image)

# Environment variables
WEAVIATE_VERSION = "1.19.6"
WEAVIATE_PORT = 8080
PERSISTENCE_PATH = "/weaviate-data"

# Create a volume for data persistence
volume = modal.Volume.from_name("weaviate-data", create_if_missing=True)

@app.function(gpu="A10G")
@modal.asgi_app()
def llm_server():
    from fastapi import FastAPI, Request
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
    import base64
    from io import BytesIO
    from PIL import Image
    import torch
    
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
    
    @fast_app.get("/health")
    async def health_check():
        return {"status": "healthy", "model": "llm_server"}
    
    return fast_app

@app.function(
    volumes={PERSISTENCE_PATH: volume},
    cpu=2.0,
    memory=4096,
)
@modal.asgi_app()
def weaviate_server():
    from fastapi import FastAPI, Request, Response
    import httpx
    import time
    import subprocess
    
    fast_app = FastAPI()
    weaviate_process = None
    
    @fast_app.on_event("startup")
    async def startup_event():
        nonlocal weaviate_process
        
        # Start Weaviate in the background
        env = os.environ.copy()
        env["AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"] = "true"
        env["PERSISTENCE_DATA_PATH"] = f"{PERSISTENCE_PATH}/data"
        env["ENABLE_MODULES"] = "text2vec-transformers"
        
        weaviate_process = subprocess.Popen(
            [
                "docker", "run",
                "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
                "-e", "PERSISTENCE_DATA_PATH=/data",
                "-e", "ENABLE_MODULES=text2vec-transformers",
                "-v", f"{PERSISTENCE_PATH}/data:/data",
                "-p", f"{WEAVIATE_PORT}:{WEAVIATE_PORT}",
                f"semitechnologies/weaviate:{WEAVIATE_VERSION}"
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Weaviate to start
        print("Starting Weaviate...")
        time.sleep(10)  # Give it time to initialize
        
        # Check if Weaviate is running
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{WEAVIATE_PORT}/v1/meta")
                if response.status_code == 200:
                    print("Weaviate is running!")
                else:
                    print(f"Weaviate startup issue: {response.status_code}")
        except Exception as e:
            print(f"Error checking Weaviate status: {str(e)}")
    
    @fast_app.on_event("shutdown")
    def shutdown_event():
        if weaviate_process:
            weaviate_process.terminate()
            weaviate_process.wait()
            print("Weaviate stopped")
    
    # Proxy all requests to Weaviate
    @fast_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    async def proxy(request: Request, path: str):
        url = f"http://localhost:{WEAVIATE_PORT}/{path}"
        
        # Forward the request to Weaviate
        async with httpx.AsyncClient() as client:
            weaviate_response = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=await request.body(),
                params=request.query_params,
                follow_redirects=True
            )
            
            # Return the response from Weaviate
            return Response(
                content=weaviate_response.content,
                status_code=weaviate_response.status_code,
                headers=dict(weaviate_response.headers)
            )
    
    @fast_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "weaviate_server"}
    
    return fast_app

@app.function(gpu="T4")
@modal.asgi_app()
def tinyllama_server():
    from fastapi import FastAPI, Request
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    fast_app = FastAPI()
    
    # Load TinyLlama model - use base model directly to avoid issues
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
    
    @fast_app.get("/health")
    async def health_check():
        return {"status": "healthy", "model": "tinyllama_server"}
    
    return fast_app

@app.function()
@modal.asgi_app()
def cloud_agent():
    from fastapi import FastAPI, Request
    from datetime import datetime
    from typing import Dict, Any, List
    
    fast_app = FastAPI()
    
    # Enhanced cloud agent with decision-making capabilities
    def evaluate_cloud_readiness(changes_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if cloud is ready for sync based on current state and incoming changes.
        """
        # Get current cloud metrics
        cloud_load = get_current_load()
        is_maintenance = is_maintenance_window()
        locked_tables = get_locked_tables()
        
        # Check if any affected tables are locked
        tables_affected = changes_data.get("tables_affected", [])
        table_conflicts = [table for table in tables_affected if table in locked_tables]
        
        # Decision logic
        if is_maintenance:
            return {
                "ready": False,
                "reason": "Maintenance window active",
                "estimated_wait": "60 minutes"
            }
        
        if table_conflicts:
            return {
                "ready": False,
                "reason": f"Tables currently locked: {', '.join(table_conflicts)}",
                "estimated_wait": "15 minutes"
            }
        
        if cloud_load > 80 and changes_data.get("priority") != "high":
            return {
                "ready": False,
                "reason": f"High system load ({cloud_load}%)",
                "estimated_wait": "5 minutes"
            }
        
        # Ready for sync
        return {
            "ready": True,
            "reason": "System ready for sync",
            "estimated_wait": "0 minutes"
        }

    def get_current_load() -> float:
        """Get current system load percentage."""
        return 50.0  # Placeholder

    def is_maintenance_window() -> bool:
        """Check if current time is in maintenance window."""
        return False  # Placeholder

    def get_locked_tables() -> List[str]:
        """Get list of currently locked tables."""
        return []  # Placeholder

    @fast_app.post("/check_cloud_state")
    async def check_cloud_state(request: Request):
        """
        Enhanced endpoint that evaluates sync requests with context.
        """
        try:
            data = await request.json()
        except:
            data = {}
        
        # Evaluate readiness with context
        assessment = evaluate_cloud_readiness(data)
        
        # Log the request
        print(f"[Cloud] Sync request received:")
        print(f"  - Priority: {data.get('priority', 'unknown')}")
        print(f"  - Tables: {data.get('tables_affected', [])}")
        print(f"  - Rows: {data.get('row_count', 0)}")
        print(f"  - Decision: {'Ready' if assessment['ready'] else 'Hold'}")
        print(f"  - Reason: {assessment['reason']}")
        
        # Return detailed response
        return {
            "status": "Ready" if assessment["ready"] else "Hold",
            "reason": assessment["reason"],
            "estimated_wait": assessment["estimated_wait"],
            "timestamp": datetime.now().isoformat()
        }
    
    @fast_app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "healthy", "service": "cloud_agent"}
    
    return fast_app

if __name__ == "__main__":
    modal.run(app)