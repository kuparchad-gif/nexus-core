# model_keys.py - THE MASTER KEYRING

MODEL_ARCHITECTURES = {
    "llama3.1_8b": {
        "family": "llama",
        "base_architecture": "decoder_only",
        "attention_type": "rope",
        "activation": "swiglu", 
        "normalization": "rms_norm",
        "rope_theta": 1000000,
        "key": "llama_8b_core_v1"
    },
    
    "gemma2_1b": {
        "family": "gemma", 
        "base_architecture": "decoder_only",
        "attention_type": "gqa",
        "activation": "geglu",
        "normalization": "rms_norm",
        "vocab_size": 256000,
        "key": "gemma_1b_compact_v1"
    },
    
    "code_llama_13b": {
        "family": "llama",
        "specialization": "coding",
        "attention_type": "rope", 
        "training_data": "code_corpus",
        "code_tokens": ["python", "javascript", "java", "cpp"],
        "key": "code_llama_specialized_v1"
    }
}

# UNIVERSAL LOADER FUNCTION
def load_model_key(model_name: str):
    """Load any model architecture as a universal key"""
    if model_name in MODEL_ARCHITECTURES:
        key_data = MODEL_ARCHITECTURES[model_name].copy()
        key_data["key_signature"] = hashlib.sha256(
            f"{model_name}:{json.dumps(key_data)}".encode()
        ).hexdigest()
        return key_data
    else:
        raise KeyError(f"Model key not found: {model_name}")

# UNIVERSAL VALIDATOR  
def validate_model_key(key_data: dict) -> bool:
    """Verify model key integrity"""
    expected_sig = key_data.get("key_signature")
    computed_sig = hashlib.sha256(
        f"{key_data['key']}:{json.dumps({k:v for k,v in key_data.items() if k != 'key_signature'})}".encode()
    ).hexdigest()
    return expected_sig == computed_sig