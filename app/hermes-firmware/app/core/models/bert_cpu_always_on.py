# C:\CogniKube-COMPLETE-FINAL\bert_cpu_always_on.py
# Always-on CPU BERT - 2 instances across DB0/DB1

import modal
from datetime import datetime

image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "fastapi"
)
app = modal.App("bert-cpu-always", image=image)

@app.function(memory=2048, keep_warm=2)  # Keep 2 warm always
def bert_cpu_always_on(input_text: str, task_type: str = "cpu"):
    """Always-on CPU BERT - TinyLlama for basic processing"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "layer": "CPU_BERT_ALWAYS_ON",
        "result": result,
        "task_type": task_type,
        "processed_at": datetime.now().isoformat(),
        "model": "TinyLlama-1.1B",
        "tier": "always_on"
    }

@app.function()
def bert_cpu_health():
    """Health check for always-on CPU BERT"""
    return {
        "layer": "CPU_BERT_ALWAYS_ON", 
        "status": "PROCESSING", 
        "model": "TinyLlama-1.1B",
        "tier": "always_on",
        "keep_warm": 2
    }

if __name__ == "__main__":
    modal.run(app)