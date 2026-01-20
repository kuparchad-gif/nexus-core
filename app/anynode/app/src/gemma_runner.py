# local_models/gemma_runner.py
# Loads Gemma-2 (base) and applies a LoRA adapter (QLoRA) if present, then runs simple generate().
import os, json
from pathlib import Path

def load_local_mind(config_path: str):
    cfg = json.loads(Path(config_path).read_text())
    model_name = cfg["model_name"]
    adapter_path = cfg.get("adapter_path")
    device = cfg.get("device","auto")
    max_new = int(cfg.get("max_new_tokens", 256))

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    except Exception as e:
        raise RuntimeError(f"Transformers not available: {e}")

    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Try 4-bit for inference if bitsandbytes exists
    quant_cfg = None
    try:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    except Exception:
        pass

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            token=hf_token,
            quantization_config=quant_cfg
        )
    except Exception:
        # fallback CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            token=hf_token
        )

    # Apply LoRA adapter if present
    if adapter_path and Path(adapter_path).exists():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as e:
            raise RuntimeError(f"PEFT adapter failed to load from {adapter_path}: {e}")

    def generate(prompt: str, system: str = None, max_new_tokens: int = None):
        text = (f"<s>[SYS]{system}[/SYS]\n{prompt}\n</s>" if system else prompt)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return tok.decode(out[0], skip_special_tokens=True)

    return {"tokenizer": tok, "model": model, "generate": generate}
