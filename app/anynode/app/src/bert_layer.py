# C:\CogniKube-COMPLETE-FINAL\bert_layer.py
# Layer 1: BERT Processing - Independent Module

import modal
from datetime import datetime
from fastapi import FastAPI, Request

image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "transformers", "torch"
)
app = modal.App("bert-layer", image=image)

@app.function(memory=2048)  # CPU only, lightweight - INTERNAL ONLY
def bert_processor(input_text: str, task_type: str = "cpu"):
    """Layer 1: CPU BERT - Internal function called by orchestrator"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load tiny model - efficient and fast
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Process with TinyLlama - cosmic power when needed
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "layer": "CPU_BERT_TINYLLAMA",
        "result": result,
        "task_type": task_type,
        "processed_at": datetime.now().isoformat(),
        "model": "TinyLlama-1.1B",
        "cosmic_power": "available"
    }

# Health check function for orchestrator to call
@app.function()
def bert_health():
    """Health check for BERT processor"""
    return {
        "layer": "CPU_BERT", 
        "status": "PROCESSING", 
        "model": "TinyLlama-1.1B", 
        "cosmic_power": "available"
    }

if __name__ == "__main__":
    modal.run(app)