# The "neuronal firing" patterns:
activation_dna = {
    "llama": {"activation": "SwiGLU", "norm": "RMSNorm"},
    "gemma": {"activation": "GeGLU", "norm": "RMSNorm"}, 
    "phi": {"activation": "ReLU", "norm": "LayerNorm"},
    "qwen": {"activation": "SwiGLU", "norm": "RMSNorm"}
}