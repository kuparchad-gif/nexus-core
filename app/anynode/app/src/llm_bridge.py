from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class NexusBridgeCore:
    def __init__(self):
        print("🧠 Loading Nexus Bridge Core LLM...")
        model_id = os.getenv("NEXUS_BRIDGE_MODEL", "google/gemma-2b-it")  # Default model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()
        print("✅ Nexus Bridge Core online.")

    def generate_response(self, prompt, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
