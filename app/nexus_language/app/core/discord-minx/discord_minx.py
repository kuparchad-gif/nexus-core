# core/voice/discord_minx.py  ← the new street core
import modal
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.11")\
    .pip_install("llama-cpp-python[server]")\
    .copy_local_file(
        "models/Discord-Micae-Hermes-3-3B.i1-Q6_K.gguf",
        "/models/discord_minx.gguf"
    )

@modal.function(cpu=4, memory=8192, timeout=3600)
async def minx_speak(prompt: str, temperature: float = 1.1) -> str:
    from llama_cpp import Llama
    
    llm = Llama(
        model_path="/models/discord_minx.gguf",
        n_ctx=8192,
        n_threads=4,
        n_gpu_layers=0,  # CPU-only forever
        verbose=False
    )
    
    output = llm(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",
        max_tokens=512,
        temperature=temperature,
        top_p=0.92,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False
    )
    return output["choices"][0]["text"].strip() + " 💀"

# Global toggle now routes through her when street = on
async def finalize_response(text: str) -> str:
    if CURRENT_VOICE_MODE == VoiceMode.STREET:
        return await minx_speak(text)  # full generation, not just translation
    return text