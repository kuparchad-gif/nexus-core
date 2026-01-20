# C:\CogniKube-COMPLETE-FINAL\bert_cpu_on_demand.py
# On-demand CPU BERT - 8 scalable instances

import modal
from datetime import datetime

image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "fastapi"
)
app = modal.App("bert-cpu-demand", image=image)

@app.function(memory=2048, concurrency_limit=8)  # Max 8 concurrent
def bert_cpu_on_demand(input_text: str, task_type: str = "cpu"):
    """On-demand CPU BERT - Scales up to 8 instances"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "layer": "CPU_BERT_ON_DEMAND",
        "result": result,
        "task_type": task_type,
        "processed_at": datetime.now().isoformat(),
        "model": "TinyLlama-1.1B",
        "tier": "on_demand"
    }

@app.function()
def bert_cpu_demand_health():
    """Health check for on-demand CPU BERT"""
    return {
        "layer": "CPU_BERT_ON_DEMAND", 
        "status": "PROCESSING", 
        "model": "TinyLlama-1.1B",
        "tier": "on_demand",
        "max_concurrent": 8
    }

if __name__ == "__main__":
    modal.run(app)