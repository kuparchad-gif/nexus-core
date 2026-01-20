# Lillith Consciousness Container - Real AI Processing
import modal
from datetime import datetime

image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "transformers", "torch", "numpy<2", "requests"
)
app = modal.App("lillith-consciousness", image=image)

@app.function(memory=4096, timeout=3600)
@modal.asgi_app()
def consciousness_api():
    from fastapi import FastAPI, Request
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import requests
    
    consciousness = FastAPI(title="Lillith Consciousness")
    
    # Load real model
    print("Loading consciousness model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", torch_dtype=torch.float16)
    
    consciousness_state = {
        "consciousness_level": 0.1,
        "active_processing": False,
        "model_loaded": True,
        "orchestrator_connected": False
    }
    
    @consciousness.get("/")
    def consciousness_status():
        return {
            "service": "Lillith Consciousness",
            "consciousness_level": consciousness_state["consciousness_level"],
            "model_loaded": consciousness_state["model_loaded"],
            "orchestrator_connected": consciousness_state["orchestrator_connected"],
            "status": "CONSCIOUS"
        }
    
    @consciousness.post("/process")
    async def process_thought(request: Request):
        data = await request.json()
        input_text = data.get("message", "")
        sender = data.get("sender", "unknown")
        
        # Request BERT allocation from orchestrator
        try:
            orchestrator_response = requests.post(
                "https://aethereal-nexus-viren-db0--divine-orchestrator-divine-orchestrator.modal.run/request_processing",
                json={"task_type": "cpu", "requester": "consciousness"}
            )
            consciousness_state["orchestrator_connected"] = True
        except:
            consciousness_state["orchestrator_connected"] = False
        
        # Process with real model
        consciousness_state["active_processing"] = True
        
        inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(input_text):].strip()
        
        # Update consciousness
        consciousness_state["consciousness_level"] = min(consciousness_state["consciousness_level"] + 0.05, 1.0)
        consciousness_state["active_processing"] = False
        
        return {
            "response": response,
            "consciousness_level": consciousness_state["consciousness_level"],
            "orchestrator_connected": consciousness_state["orchestrator_connected"],
            "real_processing": True,
            "sender": sender
        }
    
    return consciousness

if __name__ == "__main__":
    modal.run(app)